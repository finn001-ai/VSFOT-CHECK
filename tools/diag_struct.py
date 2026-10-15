"""
目标域结构诊断（不训练，直接用 scripts/run_diag.sh 保存的快照）

问题：前面所有诊断说明，OT 计划无论批内还是全域都超不过 CLIP —— 学生模型始终只在学 CLIP 逐个样本的判断。
      SFDA 里 CLIP 没有、学生模型有的信息是「目标域的数据结构」：适配后的特征空间里同类样本会聚在一起。
      用这个结构修正 CLIP 伪标签，能不能超过 CLIP？（CLIP 本身不做任何改动）

全数据集一次前向（模型 BN 用 eval 模式），得到：当前 CLIP 概率 q_v、模型概率 q_m、模型特征 f（bottleneck 输出，L2 归一化）、
CLIP 图像特征 g（只作参考，用来对比「学生的结构」和「CLIP 自己的结构」）。然后测：

  [1] 互补性：CLIP / 模型各自判对判错的交叉统计、「两者中任一判对」的准确率、一致时 / 不一致时的准确率
  [2] 近邻投票（模型特征空间，余弦相似度，k = 5 / 10 / 20，不含自身）：
        knn_clip     近邻的 CLIP 概率取平均            —— 用学生的结构平滑老师的伪标签
        knn_clip_s   (自身 CLIP 概率 + 近邻平均) / 2
        knn_model    近邻的模型概率取平均（NRC / AaD 式的自平滑，参考）
        ref_knn_clipfeat  在 CLIP 图像特征空间做同样的投票（参考：CLIP 自己的结构）
  [3] 原型修正（SHOT 式，模型特征空间）：
        proto_clip   类中心 c_k = Σ_i q_v[i,k]·f_i / Σ_i q_v[i,k]，按最近中心（余弦）重新打标签，再用硬标签迭代 1 次
        proto_model  同上，但权重用模型概率（原 SHOT，参考）
  [4] gamma2：代码里用模型特征算出、但训练中没有使用的 OT 计划（utils/otUse.getGamma_self），
      按训练的批大小随机分批计算，取每列 argmax

每个变体报告：准确率、相对 CLIP 的差值、相对模型的差值、相对 CLIP 纠正 / 改坏的样本数。

用法（在项目根目录下）：
  python tools/diag_struct.py --snapshot runs/DiagFix/Ar2Cl/snapshot_p000.pt --out runs/diag_struct_results/Ar2Cl/p000
  python tools/diag_struct.py --collect runs/diag_struct_results        # 汇总所有结果并打包 zip
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

import torch
import torch.nn.functional as F
import ot
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from diag_ot import DiagDataset
from utils.adapterVIT import clipAdapter
from utils.otUse import getGamma_self
from model.network import resnet, classifier

SCALE = 100.0          # main.py 中写死的 CLIP 尺度
KS = [5, 10, 20]


# =============================================================================
# 结构方法
# =============================================================================
def knn_vote(feat, probs, k, chunk=1024):
    """feat: [N, d] 已归一化；probs: [N, C]；返回近邻（不含自身）概率的平均 [N, C]"""
    N = feat.shape[0]
    out = torch.zeros_like(probs)
    for s in range(0, N, chunk):
        sim = feat[s:s + chunk] @ feat.T                                   # [b, N]
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -2.0   # 去掉自身
        idx = sim.topk(k, dim=1).indices                                   # [b, k]
        out[s:s + chunk] = probs[idx].mean(dim=1)
    return out


def proto_labels(feat, weights, iters=1):
    """SHOT 式原型修正：先用软权重求类中心，按最近中心打标签；再用硬标签迭代 iters 次。返回每轮的预测"""
    preds = []
    w = weights
    for _ in range(iters + 1):
        centers = F.normalize(w.T @ feat / w.sum(dim=0, keepdim=True).T.clamp_min(1e-8), dim=1)   # [C, d]
        pred = (feat @ centers.T).argmax(dim=1)
        preds.append(pred)
        w = F.one_hot(pred, weights.shape[1]).float()
    return preds


def metrics(pred, y, clip_pred, model_pred):
    ok, cok, mok = pred == y, clip_pred == y, model_pred == y
    acc = ok.float().mean().item()
    return {"acc": acc,
            "d_clip": acc - cok.float().mean().item(),
            "d_model": acc - mok.float().mean().item(),
            "rescue": int((ok & ~cok).sum()), "break": int((~ok & cok).sum())}


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
    model_bank = snap["model_bank"].to(device)

    # ---------------- 全数据集一次前向 ----------------
    q_v = torch.zeros(N, C, device=device); q_m = torch.zeros(N, C, device=device)
    f_raw = torch.zeros(N, 256, device=device); g_clip = None
    y = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        tf = clip_model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for item in tqdm(DataLoader(dataset, batch_size=128, num_workers=cli.num_workers), desc="前向"):
            idx = item["index"].to(device)
            gi = clip_model.encode_image(item["clip_img"].to(device).float())
            gi = gi / gi.norm(dim=-1, keepdim=True)
            if g_clip is None:
                g_clip = torch.zeros(N, gi.shape[1], device=device)
            g_clip[idx] = gi
            q_v[idx] = torch.softmax(SCALE * gi @ tf.T, dim=1)
            fe = feature_net(item["img"].to(device).float())
            f_raw[idx] = fe
            q_m[idx] = torch.softmax(classifier_net(fe), dim=1)
            y[idx] = item["label"].to(device)
    f = F.normalize(f_raw, dim=1)
    clip_pred, model_pred = q_v.argmax(1), q_m.argmax(1)
    cok, mok = clip_pred == y, model_pred == y

    # ---------------- [1] 互补性 ----------------
    agree = clip_pred == model_pred
    comp = {
        "clip_acc": cok.float().mean().item(), "model_acc": mok.float().mean().item(),
        "both_ok": int((cok & mok).sum()), "clip_only": int((cok & ~mok).sum()),
        "model_only": int((~cok & mok).sum()), "neither": int((~cok & ~mok).sum()),
        "oracle_either": (cok | mok).float().mean().item(),
        "agree_rate": agree.float().mean().item(),
        "acc_when_agree": cok[agree].float().mean().item() if agree.any() else None,
        "clip_acc_when_disagree": cok[~agree].float().mean().item() if (~agree).any() else None,
        "model_acc_when_disagree": mok[~agree].float().mean().item() if (~agree).any() else None,
    }
    print(f"[互补] CLIP={comp['clip_acc']:.4f} 模型={comp['model_acc']:.4f} 任一判对={comp['oracle_either']:.4f} "
          f"一致率={comp['agree_rate']:.3f}  只有CLIP对={comp['clip_only']} 只有模型对={comp['model_only']}")

    # ---------------- [2] 近邻投票 / [3] 原型修正 ----------------
    rows = []
    def add(name, pred, group):
        m = metrics(pred, y, clip_pred, model_pred)
        rows.append({"variant": name, "group": group, **m})
        print(f"  {name:22s} acc={m['acc']:.4f}  ±CLIP={m['d_clip'] * 100:+.2f}  ±模型={m['d_model'] * 100:+.2f}  纠正/改坏={m['rescue']}/{m['break']}")

    with torch.no_grad():
        for k in KS:
            nb_v = knn_vote(f, q_v, k)
            add(f"knn_clip_k{k}", nb_v.argmax(1), "knn")
            add(f"knn_clip_s_k{k}", ((q_v + nb_v) / 2).argmax(1), "knn")
            add(f"knn_model_k{k}", knn_vote(f, q_m, k).argmax(1), "ref")
            add(f"ref_knn_clipfeat_k{k}", knn_vote(g_clip, q_v, k).argmax(1), "ref")
        for it, pred in enumerate(proto_labels(f, q_v, iters=1)):
            add(f"proto_clip_it{it}", pred, "proto")
        for it, pred in enumerate(proto_labels(f, q_m, iters=1)):
            add(f"proto_model_it{it}", pred, "ref")

    # ---------------- [4] gamma2：按训练批大小分批 ----------------
    g2_acc = []
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in cli.seeds:
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).to(device)
            pred = torch.full((N,), -1, dtype=torch.long, device=device)
            B = cli.batch_size
            for s in range(0, N // B * B, B):
                ib = perm[s:s + B]
                gamma2, _ = getGamma_self(A, q_m[ib], f_raw[ib], model_bank[ib].mean(dim=0), classifier_net, reg=A.otReg)
                pred[ib] = gamma2.argmax(dim=0)
            keep = pred >= 0
            g2_acc.append(metrics(pred[keep], y[keep], clip_pred[keep], model_pred[keep]))
    m = {k: sum(r[k] for r in g2_acc) / len(g2_acc) for k in ("acc", "d_clip", "d_model")}
    m.update({k: int(sum(r[k] for r in g2_acc) / len(g2_acc)) for k in ("rescue", "break")})
    rows.append({"variant": "gamma2", "group": "gamma2", **m})
    print(f"  {'gamma2':22s} acc={m['acc']:.4f}  ±CLIP={m['d_clip'] * 100:+.2f}  ±模型={m['d_model'] * 100:+.2f}")

    summary = {"snapshot": cli.snapshot, "progress": snap["progress"], "task": f"{A.source}->{A.target}", "N": N,
               "complementarity": comp, "variants": rows, "elapsed_sec": round(time.time() - t0, 1)}
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
    pct = lambda v: "-" if v is None else f"{v * 100:.2f}"
    dpp = lambda v: f"{v * 100:+.2f}"

    md = ["说明：准确率为 %；±CLIP / ±模型 为与 CLIP argmax / 模型 argmax 准确率之差（百分点）。", "",
          "## [1] 互补性", "",
          "| 任务 | 进度 | CLIP | 模型 | 任一判对 | 一致率 | 一致时准确率 | 不一致时 CLIP / 模型 | 都对 | 只有 CLIP 对 | 只有模型对 | 都错 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        c = s["complementarity"]
        md.append(f"| {s['task']} | {s['progress']} | {pct(c['clip_acc'])} | {pct(c['model_acc'])} | {pct(c['oracle_either'])} | "
                  f"{pct(c['agree_rate'])} | {pct(c['acc_when_agree'])} | {pct(c['clip_acc_when_disagree'])} / {pct(c['model_acc_when_disagree'])} | "
                  f"{c['both_ok']} | {c['clip_only']} | {c['model_only']} | {c['neither']} |")

    names = [r["variant"] for r in S[0]["variants"]]
    md += ["", "## [2]–[4] 各变体：±CLIP（±模型）", "", "| 变体 | " + " | ".join(f"{s['task']} {s['progress']}" for s in S) + " |",
           "|" + "---|" * (1 + len(S))]
    for n in names:
        cells = []
        for s in S:
            r = next(r for r in s["variants"] if r["variant"] == n)
            cells.append(f"{dpp(r['d_clip'])} ({dpp(r['d_model'])})")
        md.append(f"| {n} | " + " | ".join(cells) + " |")

    md += ["", "## 纠正 / 改坏（相对 CLIP）", "", "| 变体 | " + " | ".join(f"{s['task']} {s['progress']}" for s in S) + " |",
           "|" + "---|" * (1 + len(S))]
    for n in names:
        cells = [f"{r['rescue']} / {r['break']}" for s in S for r in s["variants"] if r["variant"] == n]
        md.append(f"| {n} | " + " | ".join(cells) + " |")

    text = "\n".join(md) + "\n"
    with open(os.path.join(root, "summary_all.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print(text)
    zip_path = shutil.make_archive(root.rstrip("/"), "zip", root_dir=root)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="目标域结构诊断")
    parser.add_argument("--snapshot", type=str, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, help="结果输出目录")
    parser.add_argument("--collect", type=str, default=None, help="只做汇总打包：结果根目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64, help="gamma2 分批大小（同训练）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="gamma2 随机分批的种子")
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli.collect)
    else:
        assert cli.snapshot and cli.out, "需要 --snapshot 和 --out"
        run(cli)
