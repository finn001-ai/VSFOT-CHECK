"""
GLC 式全局聚类诊断：老师只负责排序，学生的特征结构负责分配（不训练，直接用 scripts/run_diag.sh 保存的快照）

动机：
  - OT 的问题在于 μ_p 来自 CLIP 校准过的概率，而 CLIP 的类先验是偏的（全域诊断：CLIP 估计的类频率与真实频率 L1 ≈ 0.23）；
  - 训练在 40%–60% 停滞，因为老师和学生 94% 的判断一致，剩下的是两边共享的错误；
  - GLC / GLC++（Qu et al., CVPR 2023 / TPAMI，arXiv 2403.14410）的一对多全局聚类：
    每类按概率取前 K 个作为正样本（只用类内排序，每类数量相同，不需要估计类先验），
    正原型 = 正样本均值，负样本 K-means 成 M 个负原型（刻画「像 c 但不是 c」），
    样本离正原型比离所有负原型都近，则标为 c；多个类满足时取离正原型最近的。

全数据集一次前向（模型 BN 用 eval 模式），CLIP 不改动。测：
  glc_clip            排序用 CLIP 概率，原型 / 分配用学生特征（K = N / C）
  glc_clip_k0.5       同上，K = N / (2C)
  glc_model           排序用学生概率（原版 GLC，参考）
  glc_clip_posonly    只用正原型：按最近正原型分配（去掉负原型，看负原型的作用）
  glc_clip_knn        glc_clip 的 one-hot 标签再做学生特征空间的局部 kNN：(自身 + 近邻平均) / 2
  knn_clip_s          参考：上一轮诊断最好的变体，(自身 CLIP + 近邻 CLIP) / 2，k = 10

指标（每个变体）：
  准确率、±CLIP、±模型、相对 CLIP 纠正 / 改坏、
  共识错误救回数：CLIP 和模型都判错的样本中，该变体判对的个数（对应训练停滞），
  共识正确改坏数：CLIP 和模型都判对的样本中，该变体判错的个数，
  预测类分布与真实类分布的 L1 距离（检验是否避开了老师的先验偏差）。

用法（在项目根目录下）：
  python tools/diag_glc.py --snapshot runs/DiagFix/Ar2Cl/snapshot_p000.pt --out runs/diag_glc_results/Ar2Cl/p000
  python tools/diag_glc.py --collect runs/diag_glc_results        # 汇总所有结果并打包 zip
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
from argparse import Namespace

import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from diag_ot import DiagDataset
from diag_struct import knn_vote
from utils.adapterVIT import clipAdapter
from model.network import resnet, classifier

SCALE = 100.0          # main.py 中写死的 CLIP 尺度
KNN_K = 10


# =============================================================================
# GLC 一对多全局聚类
# =============================================================================
def kmeans_cos(x, m, iters, seed):
    """余弦 K-means：x [n, d] 已归一化；返回 m 个归一化中心"""
    g = torch.Generator(device=x.device).manual_seed(seed)
    m = min(m, x.shape[0])
    centers = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:m]].clone()
    for _ in range(iters):
        assign = (x @ centers.T).argmax(dim=1)
        new = torch.zeros_like(centers).index_add_(0, assign, x)
        empty = torch.bincount(assign, minlength=m) == 0
        new[empty] = centers[empty]                       # 空簇保留原中心
        centers = F.normalize(new, dim=1)
    return centers


def glc_labels(f, rank_prob, k_factor=1.0, n_neg=None, iters=10, use_neg=True, seed=0):
    """
    f: [N, d] 学生特征（已归一化）；rank_prob: [N, C] 用于类内排序的概率
    返回 (预测标签 [N], 没有任何类满足条件、退回「离满足条件最近的类」的样本数)
    """
    N, C = rank_prob.shape
    K = max(1, int(N / C * k_factor))
    M = n_neg or C
    pos_sim = torch.empty(N, C, device=f.device)          # 与各类正原型的相似度
    margin = torch.empty(N, C, device=f.device)           # 正原型相似度 − 最近负原型相似度
    for c in range(C):
        top = rank_prob[:, c].topk(K).indices
        p = F.normalize(f[top].mean(dim=0), dim=0)
        pos_sim[:, c] = f @ p
        if use_neg:
            neg_mask = torch.ones(N, dtype=torch.bool, device=f.device)
            neg_mask[top] = False
            n = kmeans_cos(f[neg_mask], M, iters, seed + c)
            margin[:, c] = pos_sim[:, c] - (f @ n.T).max(dim=1).values
    if not use_neg:
        return pos_sim.argmax(dim=1), 0
    accepted = margin >= 0
    score = torch.where(accepted, pos_sim, torch.full_like(pos_sim, -1e9))
    none = ~accepted.any(dim=1)
    pred = score.argmax(dim=1)
    pred[none] = margin[none].argmax(dim=1)                # 闭集设定：没有类满足时取最接近满足的类
    return pred, int(none.sum())


def metrics(pred, y, clip_pred, model_pred, C):
    ok, cok, mok = pred == y, clip_pred == y, model_pred == y
    both_wrong, both_right = ~cok & ~mok, cok & mok
    hist = torch.bincount(pred, minlength=C).float() / len(y)
    true = torch.bincount(y, minlength=C).float() / len(y)
    acc = ok.float().mean().item()
    return {"acc": acc,
            "d_clip": acc - cok.float().mean().item(),
            "d_model": acc - mok.float().mean().item(),
            "rescue": int((ok & ~cok).sum()), "break": int((~ok & cok).sum()),
            "fix_both_wrong": int((ok & both_wrong).sum()), "n_both_wrong": int(both_wrong.sum()),
            "ruin_both_right": int((~ok & both_right).sum()), "n_both_right": int(both_right.sum()),
            "hist_l1": (hist - true).abs().sum().item()}


# =============================================================================
# 诊断
# =============================================================================
def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()

    # ---------------- 读取快照（同 diag_struct.py）----------------
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

    rows = []
    def add(name, pred, extra=None):
        m = metrics(pred, y, clip_pred, model_pred, C)
        rows.append({"variant": name, **m, **(extra or {})})
        print(f"  {name:18s} acc={m['acc']:.4f}  ±CLIP={m['d_clip'] * 100:+.2f}  ±模型={m['d_model'] * 100:+.2f}  "
              f"纠正/改坏={m['rescue']}/{m['break']}  共识错误救回={m['fix_both_wrong']}/{m['n_both_wrong']}  "
              f"共识正确改坏={m['ruin_both_right']}  类分布L1={m['hist_l1']:.3f}")

    with torch.no_grad():
        add("clip", clip_pred)
        add("model", model_pred)
        nb = knn_vote(f, q_v, KNN_K)
        add("knn_clip_s", ((q_v + nb) / 2).argmax(1))

        t = time.time()
        pred, n_none = glc_labels(f, q_v, 1.0, iters=cli.kmeans_iters)
        add("glc_clip", pred, {"n_fallback": n_none, "sec": round(time.time() - t, 1)})
        glc_clip_pred = pred
        pred, n_none = glc_labels(f, q_v, 0.5, iters=cli.kmeans_iters)
        add("glc_clip_k0.5", pred, {"n_fallback": n_none})
        pred, n_none = glc_labels(f, q_m, 1.0, iters=cli.kmeans_iters)
        add("glc_model", pred, {"n_fallback": n_none})
        pred, _ = glc_labels(f, q_v, 1.0, use_neg=False)
        add("glc_clip_posonly", pred)
        onehot = F.one_hot(glc_clip_pred, C).float()
        add("glc_clip_knn", ((onehot + knn_vote(f, onehot, KNN_K)) / 2).argmax(1))

    summary = {"snapshot": cli.snapshot, "progress": snap["progress"], "task": f"{A.source}->{A.target}", "N": N,
               "K": int(N / C), "variants": rows, "elapsed_sec": round(time.time() - t0, 1)}
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
    dpp = lambda v: f"{v * 100:+.2f}"
    names = [r["variant"] for r in S[0]["variants"]]
    cols = " | ".join(f"{s['task']} {s['progress']}" for s in S)
    sep = "|" + "---|" * (1 + len(S))
    get = lambda s, n: next(r for r in s["variants"] if r["variant"] == n)

    md = ["说明：±CLIP / ±模型 为与 CLIP argmax / 模型 argmax 准确率之差（百分点）。", "",
          "## 准确率（%）", "", f"| 变体 | {cols} |", sep]
    md += [f"| {n} | " + " | ".join(f"{get(s, n)['acc'] * 100:.2f}" for s in S) + " |" for n in names]
    md += ["", "## ±CLIP（±模型）", "", f"| 变体 | {cols} |", sep]
    md += [f"| {n} | " + " | ".join(f"{dpp(get(s, n)['d_clip'])} ({dpp(get(s, n)['d_model'])})" for s in S) + " |" for n in names]
    md += ["", "## 共识错误救回 / 共识正确改坏（两边都错的样本中判对的个数 / 两边都对的样本中判错的个数）", "",
           f"| 变体 | {cols} |", sep,
           "| （两边都错 / 两边都对的总数） | " + " | ".join(f"{get(s, 'clip')['n_both_wrong']} / {get(s, 'clip')['n_both_right']}" for s in S) + " |"]
    md += [f"| {n} | " + " | ".join(f"{get(s, n)['fix_both_wrong']} / {get(s, n)['ruin_both_right']}" for s in S) + " |" for n in names]
    md += ["", "## 纠正 / 改坏（相对 CLIP）", "", f"| 变体 | {cols} |", sep]
    md += [f"| {n} | " + " | ".join(f"{get(s, n)['rescue']} / {get(s, n)['break']}" for s in S) + " |" for n in names]
    md += ["", "## 预测类分布与真实类分布的 L1 距离", "", f"| 变体 | {cols} |", sep]
    md += [f"| {n} | " + " | ".join(f"{get(s, n)['hist_l1']:.3f}" for s in S) + " |" for n in names]
    md += ["", "## GLC 退回样本数（没有任何类满足「离正原型比离负原型近」）", "", f"| 变体 | {cols} |", sep]
    md += [f"| {n} | " + " | ".join(str(get(s, n).get("n_fallback", "-")) for s in S) + " |"
           for n in names if "n_fallback" in S[0]["variants"][names.index(n)]]

    text = "\n".join(md) + "\n"
    with open(os.path.join(root, "summary_all.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print(text)
    zip_path = shutil.make_archive(root.rstrip("/"), "zip", root_dir=root)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLC 式全局聚类诊断")
    parser.add_argument("--snapshot", type=str, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, help="结果输出目录")
    parser.add_argument("--collect", type=str, default=None, help="只做汇总打包：结果根目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kmeans_iters", type=int, default=10, help="负样本 K-means 迭代次数")
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli.collect)
    else:
        assert cli.snapshot and cli.out, "需要 --snapshot 和 --out"
        run(cli)
