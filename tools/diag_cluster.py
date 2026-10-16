"""
簇级命名诊断（方案 A）：学生分组，CLIP 命名（不训练，直接用 scripts/run_diag.sh 保存的快照）

问题：标签层面的所有改进（OT、近邻平滑、裁决）都被「CLIP 和学生任一判对」的并集卡住（Ar→Cl 约 76%）。
      换一种分工：学生用自己的特征把目标域分成很多小簇，CLIP 只负责给每个簇命名（簇内投票），
      CLIP 在单个样本上的错误会被簇内多数纠正。此时上限变成「学生分簇的纯度」，而不是 CLIP 的准确率。
      要回答：这样能不能明显越过并集上限？

全数据集一次前向（模型 BN 用 eval 模式），得到当前 CLIP 概率 q_v、模型概率 q_m、学生特征 f（L2 归一化）、标签 y。
对每个 K（簇数）：
  - 在 f 上做球面 k-means（k-means++ 初始化，跑 3 个种子，按目标函数选最好的，不用标签）
  - 纯度（用了标签，上限参考）：每个簇按真实多数类命名时的准确率
  - 命名方式（都不用标签）：
      vote_hard    簇内 CLIP argmax 的多数票
      vote_soft    簇内 CLIP 概率求和后取 argmax
      ot_clipmean  OT 求「簇 → 类」分配：代价 −log(簇内平均 CLIP 概率)，簇边际 = 簇大小，类边际 = 全域 CLIP 均值
      ot_uniform   同上，类边际 = 均匀
      ot_true      同上，类边际 = 真实全域类频率（用了标签，参考）
      mix_soft     (自身 CLIP 概率 + 所在簇的平均 CLIP 概率) / 2，逐样本取 argmax
    第二个命名员（错误模式和 CLIP 不同；2026-10-16 加入，见 archive/2026_10_16_diag_cluster）：
      vote_model   簇内学生概率求和后取 argmax
      vote_prod    簇内 CLIP 平均概率 × 簇内学生平均概率（对数相加）取 argmax：两个专家都认可的命名才胜出
      ot_prod      OT 簇 → 类：代价 −log(簇内 CLIP 均值) − log(簇内学生均值)，类边际 = (CLIP 全域均值 + 学生全域均值) / 2
      mix_prod     (自身 CLIP + 自身学生) / 2 与所在簇的乘积命名概率再平均，逐样本取 argmax
  - 参考：底座里的近邻平滑 knn_clip_s（k = 10）
指标：准确率、相对 CLIP / 模型 / 「任一判对」并集的差值、纠正 / 改坏数，
      以及 **两边都错却被救回的样本数**（CLIP 和学生都判错、簇级命名判对）—— 只有这一项能越过并集上限。

用法（在项目根目录下）：
  python tools/diag_cluster.py --snapshot runs/DiagFix/Ar2Cl/snapshot_p050.pt --out runs/diag_cluster_results/Ar2Cl/p050
  python tools/diag_cluster.py --collect runs/diag_cluster_results        # 汇总所有结果并打包 zip
"""
import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import time
import shutil
import argparse
import warnings
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
import ot
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from diag_ot import DiagDataset
from diag_struct import knn_vote
from utils.adapterVIT import clipAdapter
from model.network import resnet, classifier

SCALE = 100.0          # main.py 中写死的 CLIP 尺度
NAMINGS = ["vote_hard", "vote_soft", "ot_clipmean", "ot_uniform", "ot_true", "mix_soft",
           "vote_model", "vote_prod", "ot_prod", "mix_prod"]


# =============================================================================
# 球面 k-means
# =============================================================================
def spherical_kmeans(x, K, iters, seed):
    """x: [N, d] 已归一化。k-means++ 初始化；返回 (每个样本的簇号, 目标函数 = Σ 与所属中心的余弦相似度)"""
    N = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    centers = [x[torch.randint(N, (1,), generator=g).item()]]
    d2 = torch.full((N,), float("inf"), device=x.device)
    for _ in range(1, K):                                   # k-means++：按与已选中心的距离平方加权抽样
        d2 = torch.minimum(d2, (2 - 2 * x @ centers[-1]).clamp_min(0))
        p = (d2 / d2.sum()).cpu()
        centers.append(x[torch.multinomial(p, 1, generator=g).item()])
    C = torch.stack(centers)
    for _ in range(iters):
        assign = (x @ C.T).argmax(dim=1)
        newC = torch.zeros_like(C).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=K)
        empty = cnt == 0
        if empty.any():                                     # 空簇：随机重新放一个样本点
            newC[empty] = x[torch.randint(N, (int(empty.sum()),), generator=g).to(x.device)]
        C = F.normalize(newC, dim=1)
    sim = x @ C.T
    assign = sim.argmax(dim=1)
    return assign, float(sim.max(dim=1).values.sum())


# =============================================================================
# 簇级命名
# =============================================================================
def ot_name(mean_cost, size, N, mu, reg, K, device):
    """OT 簇 → 类：mean_cost [K, C]，簇边际 = 簇大小 / N，类边际 = mu；返回每个簇的类"""
    keep = size > 0
    a = (size[keep] / N).double().cpu().numpy()
    M = mean_cost[keep].double().cpu().numpy()
    b = np.clip(mu, 1e-12, None); b = b / b.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = ot.sinkhorn(a, b, M, reg, method="sinkhorn_log", numItermax=5000, stopThr=1e-9)
    label = torch.zeros(K, dtype=torch.long, device=device)
    label[keep] = torch.from_numpy(plan.argmax(axis=1)).to(device)
    return label


def name_clusters(assign, K, q_v, mus, reg, q_m=None):
    """返回 {命名方式: 每个样本的预测类}"""
    N, C = q_v.shape
    onehot = F.one_hot(assign, K).float()                   # [N, K]
    size = onehot.sum(dim=0)                                # [K]
    mean_p = (onehot.T @ q_v) / size.clamp_min(1).unsqueeze(1)       # [K, C] 簇内平均 CLIP 概率
    hard_votes = onehot.T @ F.one_hot(q_v.argmax(1), C).float()      # [K, C] 簇内 argmax 票数
    out = {
        "vote_hard": hard_votes.argmax(1)[assign],
        "vote_soft": mean_p.argmax(1)[assign],
        "mix_soft": ((q_v + mean_p[assign]) / 2).argmax(1),
    }
    # OT：簇 → 类，代价 −log(簇内平均概率)，簇边际 = 簇大小 / N
    keep = size > 0
    a = (size[keep] / N).double().cpu().numpy()
    M = (-torch.log(mean_p[keep] + 1e-6)).double().cpu().numpy()
    for name, mu in mus.items():
        b = np.clip(mu, 1e-12, None); b = b / b.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = ot.sinkhorn(a, b, M, reg, method="sinkhorn_log", numItermax=5000, stopThr=1e-9)
        label = torch.full((K,), 0, dtype=torch.long, device=q_v.device)
        label[keep] = torch.from_numpy(plan.argmax(axis=1)).to(q_v.device)
        out[name] = label[assign]
    if q_m is not None:
        mean_m = (onehot.T @ q_m) / size.clamp_min(1).unsqueeze(1)                  # [K, C] 簇内平均学生概率
        log_prod = torch.log(mean_p + 1e-6) + torch.log(mean_m + 1e-6)             # 两个专家的乘积（对数）
        prod = torch.softmax(log_prod, dim=1)
        out["vote_model"] = mean_m.argmax(1)[assign]
        out["vote_prod"] = log_prod.argmax(1)[assign]
        mu_avg = (q_v.mean(0) + q_m.mean(0)).cpu().numpy() / 2
        out["ot_prod"] = ot_name(-log_prod, size, N, mu_avg, reg, K, q_v.device)[assign]
        out["mix_prod"] = (((q_v + q_m) / 2 + prod[assign]) / 2).argmax(1)
    return out


def metrics(pred, y, clip_pred, model_pred):
    ok, cok, mok = pred == y, clip_pred == y, model_pred == y
    acc = ok.float().mean().item()
    return {"acc": acc,
            "d_clip": acc - cok.float().mean().item(),
            "d_model": acc - mok.float().mean().item(),
            "d_union": acc - (cok | mok).float().mean().item(),
            "rescue": int((ok & ~cok).sum()), "break": int((~ok & cok).sum()),
            "both_wrong_saved": int((ok & ~cok & ~mok).sum())}


# =============================================================================
# 诊断
# =============================================================================
def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()

    # ---------------- 读取快照（同 diag_ot.py）----------------
    snap = torch.load(cli.snapshot, map_location="cpu", weights_only=False)
    A = Namespace(**snap["args"])
    A.device = device
    C = A.class_num
    print(f"[快照] {cli.snapshot}  progress={snap['progress']}  iter={snap['iter']}/{snap['max_iter']}")

    raw_clip, preprocess = clip.load(A.clipVersion, device=device)
    clip_model = clipAdapter(raw_clip, device, adapterCount=A.adaptercount)
    missing, unexpected = clip_model.load_state_dict(snap["adapter"], strict=False)
    assert not unexpected and not [k for k in missing if "adapter" in k], "adapter 参数载入不完整"
    clip_model.eval()
    feature_net = resnet(res_name=A.arch).to(device)
    feature_net.load_state_dict(snap["feature"])
    feature_net.eval()
    classifier_net = classifier(class_num=C).to(device)
    classifier_net.load_state_dict(snap["classifier"])
    classifier_net.eval()

    dataset = DiagDataset(snap["imgs"], preprocess, A.class_info, isTraining=False)
    N = len(dataset)

    # ---------------- 全数据集一次前向 ----------------
    q_v = torch.zeros(N, C, device=device); q_m = torch.zeros(N, C, device=device)
    f = torch.zeros(N, 256, device=device); y = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        tf = clip_model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for item in tqdm(DataLoader(dataset, batch_size=128, num_workers=cli.num_workers), desc="前向"):
            idx = item["index"].to(device)
            gi = clip_model.encode_image(item["clip_img"].to(device).float())
            gi = gi / gi.norm(dim=-1, keepdim=True)
            q_v[idx] = torch.softmax(SCALE * gi @ tf.T, dim=1)
            fe = feature_net(item["img"].to(device).float())
            f[idx] = F.normalize(fe, dim=1)
            q_m[idx] = torch.softmax(classifier_net(fe), dim=1)
            y[idx] = item["label"].to(device)
    clip_pred, model_pred = q_v.argmax(1), q_m.argmax(1)
    yn = y.cpu().numpy()
    mus = {"ot_clipmean": q_v.mean(0).cpu().numpy(),
           "ot_uniform": np.full(C, 1.0 / C),
           "ot_true": np.bincount(yn, minlength=C) / N}
    ref = {"N": N, "clip_acc": (clip_pred == y).float().mean().item(), "model_acc": (model_pred == y).float().mean().item(),
           "union_acc": ((clip_pred == y) | (model_pred == y)).float().mean().item(),
           "both_wrong": int(((clip_pred != y) & (model_pred != y)).sum())}
    with torch.no_grad():
        nb = knn_vote(f, q_v, 10)
    ref["knn_clip_s_k10"] = metrics(((q_v + nb) / 2).argmax(1), y, clip_pred, model_pred)
    print(f"[参考] CLIP={ref['clip_acc']:.4f}  模型={ref['model_acc']:.4f}  并集={ref['union_acc']:.4f}  两边都错={ref['both_wrong']}  "
          f"knn_clip_s={ref['knn_clip_s_k10']['acc']:.4f}")

    # ---------------- 各 K：聚类 → 纯度 → 命名 ----------------
    rows = []
    with torch.no_grad():
        for K in cli.ks:
            best = max((spherical_kmeans(f, K, cli.kmeans_iters, s) for s in range(cli.kmeans_seeds)), key=lambda r: r[1])
            assign = best[0]
            onehot_y = F.one_hot(assign, K).float().T @ F.one_hot(y, C).float()       # [K, C] 簇内真实类计数
            purity = onehot_y.max(dim=1).values.sum().item() / N
            sizes = torch.bincount(assign, minlength=K)
            row = {"K": K, "purity": purity, "n_empty": int((sizes == 0).sum()), "min_size": int(sizes[sizes > 0].min()),
                   "max_size": int(sizes.max())}
            for name, pred in name_clusters(assign, K, q_v, mus, cli.reg, q_m).items():
                row[name] = metrics(pred, y, clip_pred, model_pred)
            rows.append(row)
            print(f"  K={K:4d} 纯度={purity:.4f}  " + "  ".join(
                f"{n}={row[n]['acc']:.4f}({row[n]['d_clip'] * 100:+.1f}, 救回{row[n]['both_wrong_saved']})" for n in NAMINGS))

    summary = {"snapshot": cli.snapshot, "progress": snap["progress"], "task": f"{A.source}->{A.target}",
               "ref": ref, "rows": rows, "reg": cli.reg, "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(cli.out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {cli.out}（用时 {summary['elapsed_sec']}s）")


# =============================================================================
# 汇总所有结果并打包
# =============================================================================
def collect(root):
    files = sorted(glob.glob(os.path.join(root, "*", "*", "summary.json")))
    if not files:
        print(f"[错误] {root} 下没有找到 summary.json"); return
    S = [json.load(open(fp, encoding="utf-8")) for fp in files]
    pct = lambda v: f"{v * 100:.2f}"
    dpp = lambda v: f"{v * 100:+.2f}"

    md = ["说明：准确率为 %。±CLIP / ±并集 为相对 CLIP argmax / 「CLIP 和学生任一判对」的百分点；",
          "「救回」= CLIP 和学生都判错、该方法判对的样本数（只有它能越过并集上限）。", "",
          "## 参考", "", "| 任务 | 进度 | CLIP | 模型 | 并集 | 两边都错 | knn_clip_s（k=10）±CLIP | 其救回数 |",
          "|---|---|---|---|---|---|---|---|"]
    for s in S:
        r = s["ref"]
        md.append(f"| {s['task']} | {s['progress']} | {pct(r['clip_acc'])} | {pct(r['model_acc'])} | {pct(r['union_acc'])} | "
                  f"{r['both_wrong']} | {dpp(r['knn_clip_s_k10']['d_clip'])} | {r['knn_clip_s_k10']['both_wrong_saved']} |")

    md += ["", "## 簇纯度（用了标签，上限参考）", "", "| 任务 | 进度 | " + " | ".join(f"K={r['K']}" for r in S[0]["rows"]) + " |",
           "|" + "---|" * (2 + len(S[0]["rows"]))]
    for s in S:
        md.append(f"| {s['task']} | {s['progress']} | " + " | ".join(pct(r["purity"]) for r in s["rows"]) + " |")

    for name in NAMINGS:
        md += ["", f"## {name}：准确率（±CLIP / ±并集 / 救回）", "", "| 任务 | 进度 | " + " | ".join(f"K={r['K']}" for r in S[0]["rows"]) + " |",
               "|" + "---|" * (2 + len(S[0]["rows"]))]
        for s in S:
            md.append(f"| {s['task']} | {s['progress']} | " + " | ".join(
                f"{pct(r[name]['acc'])}（{dpp(r[name]['d_clip'])} / {dpp(r[name]['d_union'])} / {r[name]['both_wrong_saved']}）"
                for r in s["rows"]) + " |")

    text = "\n".join(md) + "\n"
    with open(os.path.join(root, "summary_all.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print(text)
    zip_path = shutil.make_archive(root.rstrip("/"), "zip", root_dir=root)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="簇级命名诊断：学生分组，CLIP 命名")
    parser.add_argument("--snapshot", type=str, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, help="结果输出目录")
    parser.add_argument("--collect", type=str, default=None, help="只做汇总打包：结果根目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ks", type=int, nargs="+", default=[65, 130, 260, 520], help="簇数")
    parser.add_argument("--kmeans_iters", type=int, default=50)
    parser.add_argument("--kmeans_seeds", type=int, default=3, help="k-means 跑几个种子，按目标函数选最好的（不用标签）")
    parser.add_argument("--reg", type=float, default=0.05, help="簇 → 类 OT 的熵正则 ε")
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli.collect)
    else:
        assert cli.snapshot and cli.out, "需要 --snapshot 和 --out"
        run(cli)
