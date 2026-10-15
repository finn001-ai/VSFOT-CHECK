"""
全数据集 OT 诊断（不训练，直接用 scripts/run_diag.sh 保存的快照）

问题：批内（64 张）OT 的类别边际 μ_p 必须针对本批，而批内类分布和全域差别很大，所以全域频率在批内很差，
      CLIP 推出的批内 μ_p 又只能复述 CLIP。如果在整个目标域上做一次 OT，用全域类边际，
      就不存在「批内与全域不一致」的问题，而且全域边际不用标签也能估得比较准。

做三件事（代价矩阵公式和 utils/otUse.getGamma 完全相同：α·(1 − p) − log(p + 1e-6)，α = 1 / max(1 − p)）：
  [1] 全域 OT：对全部 N 张图做一次 Sinkhorn（类别边际 = 全域 μ，样本边际 = 1/N），换不同的 μ：
        true_global      真实全域类频率（上界，用了标签）
        clip_mean        当前 CLIP 概率的全域均值
        bank_mean        记忆库 CLIP 概率的全域均值（训练中实际能拿到的）
        model_mean       模型概率的全域均值
        clip_model_avg   (clip_mean + model_mean) / 2
        clip_sharp_mean  CLIP 概率逐行 p^(1/0.25) 归一化后取全域均值
        em_prior         标签偏移的 EM 先验估计（Saerens 等）：π ← mean_i[p_ik π_k / Σ_l p_il π_l]，迭代 20 次
        uniform          均匀分布
      核（代价矩阵所用的 CLIP 概率）分两种：clip_now（当前 CLIP）和 bank（记忆库，训练中每个 epoch 能用的）
  [2] 规模曲线：把数据随机切成大小为 n 的子集（n = 64 / 256 / 1024 / N）分别做 OT，类别边际取全域 μ，
      看准确率是否随 n 上升（n = 64 时即「批内 OT + 全域边际」）
  [3] ε 网格：0.1 / 0.2 / 0.4（核 = clip_now）

求解：ot.sinkhorn(method="sinkhorn_log")，float64；记录行 / 列边际误差。
指标：计划 argmax 准确率（每个样本取计划列的 argmax）、相对 CLIP argmax 的差值、真类质量、
      μ 与真实全域频率的 L1 距离、相对 CLIP argmax 纠正 / 改坏的样本数。
模型概率用 BN eval 模式（全数据集一次前向，不依赖批）。

用法（在项目根目录下）：
  python tools/diag_global_ot.py --snapshot runs/DiagFix/Ar2Cl/snapshot_p000.pt --out runs/diag_global_results/Ar2Cl/p000
  python tools/diag_global_ot.py --collect runs/diag_global_results        # 汇总所有结果并打包 zip
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
import ot
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from diag_ot import DiagDataset
from utils.adapterVIT import clipAdapter
from model.network import resnet, classifier

SCALE = 100.0          # main.py 中写死的 CLIP 尺度
SHARP_T = 0.25
EM_ITERS = 20
MU_NAMES = ["true_global", "clip_mean", "bank_mean", "model_mean", "clip_model_avg", "clip_sharp_mean", "em_prior", "uniform"]


# =============================================================================
# OT 计算
# =============================================================================
def ot_cost(p):
    """p: [n, C] 概率 -> 代价 [C, n]，公式同 getGamma"""
    feat = 1.0 - p.T
    return feat / feat.max() - np.log(p.T + 1e-6)


def solve(p, mu, reg, num_iter):
    """全域 / 子集 OT：类别边际 mu [C]，样本边际 1/n；返回计划 [C, n] 和边际误差"""
    n = p.shape[0]
    mu = np.clip(mu, 1e-12, None)
    mu = mu / mu.sum()
    nu = np.full(n, 1.0 / n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ot.sinkhorn(mu, nu, ot_cost(p), reg, method="sinkhorn_log", numItermax=num_iter, stopThr=1e-9)
    row_err = float(np.abs(g.sum(axis=1) - mu).sum())
    col_err = float(np.abs(g.sum(axis=0) - nu).sum())
    return g, row_err, col_err


def plan_metrics(g, y, clip_pred):
    """g: [C, n]，y / clip_pred: [n]"""
    pred = g.argmax(axis=0)
    ok, clip_ok = pred == y, clip_pred == y
    return {
        "plan_acc": float(ok.mean()),
        "true_mass": float(g[y, np.arange(len(y))].sum() / g.sum()),
        "rescue": int((ok & ~clip_ok).sum()),     # CLIP 判错、计划判对
        "break": int((~ok & clip_ok).sum()),      # CLIP 判对、计划判错
    }


def em_prior(p, iters):
    """标签偏移 EM：假设 CLIP 的隐含先验为均匀分布，估计目标域类先验"""
    pi = np.full(p.shape[1], 1.0 / p.shape[1])
    for _ in range(iters):
        q = p * pi
        q = q / q.sum(axis=1, keepdims=True)
        pi = q.mean(axis=0)
    return pi


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
    bank = snap["vil_bank"].double().numpy()                    # [N, C] 记忆库 CLIP 概率

    # ---------------- 全数据集一次前向：当前 CLIP 概率、模型概率、标签 ----------------
    clip_p = np.zeros((N, C)); model_p = np.zeros((N, C)); y = np.zeros(N, dtype=np.int64)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        tf = clip_model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for item in tqdm(DataLoader(dataset, batch_size=128, num_workers=cli.num_workers), desc="前向"):
            idx = item["index"].numpy()
            f = clip_model.encode_image(item["clip_img"].to(device).float())
            f = f / f.norm(dim=-1, keepdim=True)
            clip_p[idx] = torch.softmax(SCALE * f @ tf.T, dim=1).double().cpu().numpy()
            model_p[idx] = torch.softmax(classifier_net(feature_net(item["img"].to(device).float())), dim=1).double().cpu().numpy()
            y[idx] = item["label"].numpy()

    clip_pred = clip_p.argmax(1)
    true_global = np.bincount(y, minlength=C) / N
    sharp = clip_p ** (1.0 / SHARP_T)
    sharp = sharp / sharp.sum(axis=1, keepdims=True)
    mus = {
        "true_global": true_global,
        "clip_mean": clip_p.mean(0),
        "bank_mean": bank.mean(0),
        "model_mean": model_p.mean(0),
        "clip_model_avg": (clip_p.mean(0) + model_p.mean(0)) / 2,
        "clip_sharp_mean": sharp.mean(0),
        "em_prior": em_prior(clip_p, EM_ITERS),
        "uniform": np.full(C, 1.0 / C),
    }
    ref = {
        "N": N, "clip_acc": float((clip_pred == y).mean()), "model_acc": float((model_p.argmax(1) == y).mean()),
        "bank_acc": float((bank.argmax(1) == y).mean()),
        "mu_l1": {k: float(np.abs(v / v.sum() - true_global).sum()) for k, v in mus.items()},
    }
    print(f"[参考] N={N}  CLIP={ref['clip_acc']:.4f}  模型={ref['model_acc']:.4f}  记忆库={ref['bank_acc']:.4f}")

    # ---------------- [1] 全域 OT：核 × μ ----------------
    full = []
    for kernel, p in [("clip_now", clip_p), ("bank", bank)]:
        for name in MU_NAMES:
            g, re, ce = solve(p, mus[name], A.otReg, cli.num_iter)
            m = plan_metrics(g, y, clip_pred)
            full.append({"kernel": kernel, "mu": name, "eps": A.otReg, **m,
                         "delta_vs_clip": m["plan_acc"] - ref["clip_acc"], "mu_l1": ref["mu_l1"][name],
                         "row_err": re, "col_err": ce})
            print(f"  [全域] 核={kernel:8s} μ={name:15s} acc={m['plan_acc']:.4f} ({(m['plan_acc'] - ref['clip_acc']) * 100:+.2f})"
                  f"  纠正/改坏={m['rescue']}/{m['break']}  列误差={ce:.1e}")

    # ---------------- [2] 规模曲线：子集大小 n，类别边际 = 全域 μ ----------------
    curve = []
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    for n in cli.sizes + [N]:
        n = min(n, N)
        chunks = [perm[i:i + n] for i in range(0, N - n + 1, n)]      # 丢弃不满 n 的尾部
        for name in cli.curve_mus:
            accs, tm = [], []
            for ch in chunks:
                g, _, _ = solve(clip_p[ch], mus[name], A.otReg, cli.num_iter)
                m = plan_metrics(g, y[ch], clip_pred[ch])
                accs.append(m["plan_acc"] * len(ch)); tm.append(m["true_mass"] * len(ch))
            covered = sum(len(c) for c in chunks)
            acc = sum(accs) / covered
            clip_acc_cov = float((clip_pred[np.concatenate(chunks)] == y[np.concatenate(chunks)]).mean())
            curve.append({"n": n, "mu": name, "n_chunks": len(chunks), "plan_acc": acc,
                          "true_mass": sum(tm) / covered, "delta_vs_clip": acc - clip_acc_cov})
            print(f"  [规模] n={n:5d} μ={name:15s} acc={acc:.4f} ({(acc - clip_acc_cov) * 100:+.2f})  子集数={len(chunks)}")

    # ---------------- [3] ε 网格（核 = clip_now）----------------
    eps_rows = []
    for eps in cli.regs:
        for name in cli.curve_mus:
            g, re, ce = solve(clip_p, mus[name], eps, cli.num_iter)
            m = plan_metrics(g, y, clip_pred)
            eps_rows.append({"eps": eps, "mu": name, **m, "delta_vs_clip": m["plan_acc"] - ref["clip_acc"], "col_err": ce})

    summary = {"snapshot": cli.snapshot, "progress": snap["progress"], "task": f"{A.source}->{A.target}",
               "otReg": A.otReg, "ref": ref, "full": full, "curve": curve, "eps": eps_rows,
               "versions": {"torch": torch.__version__, "pot": ot.__version__},
               "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(cli.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {cli.out}（用时 {summary['elapsed_sec']}s）")


# =============================================================================
# 汇总所有结果并打包
# =============================================================================
def collect(root):
    files = sorted(glob.glob(os.path.join(root, "*", "*", "summary.json")))
    if not files:
        print(f"[错误] {root} 下没有找到 summary.json"); return
    S = [json.load(open(f, encoding="utf-8")) for f in files]
    pct = lambda v: f"{v * 100:.2f}"
    dpp = lambda v: f"{v * 100:+.2f}"

    md = ["说明：准确率为 %；「±CLIP」= 计划 argmax 准确率 − CLIP argmax 准确率（百分点）。", ""]
    md += ["## [1] 全域 OT（ε = otReg）", "",
           "| 任务 | 进度 | CLIP | 模型 | 核 | " + " | ".join(MU_NAMES) + " |", "|" + "---|" * (5 + len(MU_NAMES))]
    for s in S:
        for kernel in ("clip_now", "bank"):
            cells = []
            for name in MU_NAMES:
                r = next(r for r in s["full"] if r["kernel"] == kernel and r["mu"] == name)
                cells.append(f"{pct(r['plan_acc'])} ({dpp(r['delta_vs_clip'])})")
            md.append(f"| {s['task']} | {s['progress']} | {pct(s['ref']['clip_acc'])} | {pct(s['ref']['model_acc'])} | {kernel} | "
                      + " | ".join(cells) + " |")

    md += ["", "## μ 与真实全域频率的 L1 距离", "", "| 任务 | 进度 | " + " | ".join(MU_NAMES) + " |", "|" + "---|" * (2 + len(MU_NAMES))]
    for s in S:
        md.append(f"| {s['task']} | {s['progress']} | " + " | ".join(f"{s['ref']['mu_l1'][n]:.3f}" for n in MU_NAMES) + " |")

    md += ["", "## 全域 OT 纠正 / 改坏（核 = clip_now）", "", "| 任务 | 进度 | " + " | ".join(MU_NAMES) + " |", "|" + "---|" * (2 + len(MU_NAMES))]
    for s in S:
        cells = [f"{r['rescue']} / {r['break']}" for n in MU_NAMES for r in s["full"] if r["kernel"] == "clip_now" and r["mu"] == n]
        md.append(f"| {s['task']} | {s['progress']} | " + " | ".join(cells) + " |")

    curve_mus = sorted({r["mu"] for s in S for r in s["curve"]}, key=lambda m: MU_NAMES.index(m))
    # 最后一列是全量 N，各任务的 N 不同，所以表头写「全量」
    sizes = sorted({r["n"] for r in S[0]["curve"]})[:-1]
    md += ["", "## [2] 规模曲线（核 = clip_now，类别边际 = 全域 μ；单元格为 ±CLIP）", "",
           "| 任务 | 进度 | μ | " + " | ".join(f"n={n}" for n in sizes) + " | 全量 |",
           "|" + "---|" * (4 + len(sizes))]
    for s in S:
        ns = sorted({r["n"] for r in s["curve"]})
        for name in curve_mus:
            cells = [dpp(next(r for r in s["curve"] if r["n"] == n and r["mu"] == name)["delta_vs_clip"]) for n in ns]
            md.append(f"| {s['task']} | {s['progress']} | {name} | " + " | ".join(cells) + " |")

    regs = sorted({r["eps"] for s in S for r in s["eps"]})
    md += ["", "## [3] ε 网格（全域，核 = clip_now；单元格为 ±CLIP）", "",
           "| 任务 | 进度 | μ | " + " | ".join(f"ε={e:g}" for e in regs) + " |", "|" + "---|" * (3 + len(regs))]
    for s in S:
        for name in curve_mus:
            cells = [dpp(next(r for r in s["eps"] if r["eps"] == e and r["mu"] == name)["delta_vs_clip"]) for e in regs]
            md.append(f"| {s['task']} | {s['progress']} | {name} | " + " | ".join(cells) + " |")

    md += ["", "## 收敛检查（全域 OT 的最大列边际误差）", "", "| 任务 | 进度 | 最大列误差 | 最大行误差 |", "|---|---|---|---|"]
    for s in S:
        md.append(f"| {s['task']} | {s['progress']} | {max(r['col_err'] for r in s['full']):.1e} | {max(r['row_err'] for r in s['full']):.1e} |")

    text = "\n".join(md) + "\n"
    with open(os.path.join(root, "summary_all.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    zip_path = shutil.make_archive(root.rstrip("/"), "zip", root_dir=root)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全数据集 OT 诊断")
    parser.add_argument("--snapshot", type=str, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, help="结果输出目录")
    parser.add_argument("--collect", type=str, default=None, help="只做汇总打包：结果根目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024], help="规模曲线的子集大小（全量 N 自动加入）")
    parser.add_argument("--curve_mus", type=str, nargs="+", default=["true_global", "clip_mean", "em_prior"],
                        help="规模曲线 / ε 网格要测的 μ")
    parser.add_argument("--regs", type=float, nargs="+", default=[0.1, 0.2, 0.4], help="ε 网格")
    parser.add_argument("--num_iter", type=int, default=5000, help="Sinkhorn 最大迭代次数")
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli.collect)
    else:
        assert cli.snapshot and cli.out, "需要 --snapshot 和 --out"
        run(cli)
