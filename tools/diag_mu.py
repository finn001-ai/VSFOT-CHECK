"""
μ_p 估计诊断（不训练，直接用 scripts/run_diag.sh 保存的快照）

问题：Sinkhorn 的类别边际 μ_p 是 OT 计划里真正起作用的部分。oracle μ_p（本批真实类直方图）比 CLIP 高 7–12pp，
      不用标签的估计器能把 bank → true_batch 之间的差距（headroom）收回多少？

每批用同一份 CLIP 概率、不改动的 utils/otUse.getGamma(A, clip_prob.T, μ, A.otReg)，只换 μ_p：
  bank            记忆库 CLIP 概率的批均值（main.py 实际用的）
  true_batch      本批真实类直方图（oracle 上界）
  uniform         均匀分布
  clip_now        当前 CLIP 概率的批均值（参考）
  model           模型概率的批均值（BN 用 train 模式 / 批统计量，同 diag_quick）
  clip_model_avg  (clip_now + model) / 2
  sharp_T{T}      CLIP 概率逐行做 p^(1/T) 再归一化后取批均值，T ∈ {0.5, 0.25}
  present_tau{τ}  bank 中「批内 CLIP 最大概率 < τ」的类置 0 再归一化，τ ∈ {0.3, 0.5}；全被置 0 时退回 bank
  em_rowsum{k}    μ ← 用 μ 算出的计划的行和，从 bank 出发迭代 k ∈ {1, 3} 次
                  注意：平衡 Sinkhorn 收敛后行和 ≡ μ，所以这一项理论上 ≈ bank，只作健全性检查
  em_mix{k}       非退化的自洽版本：把 CLIP 概率当作各类似然，做混合比例 EM
                  μ_k ← mean_i [ p_ik μ_k / Σ_l p_il μ_l ]，从 bank 出发迭代 k ∈ {1, 3} 次，再用 getGamma 算计划
  uot_m{ρ}        半松弛 OT：类别边际用 KL 松弛（ρ ∈ {0.1, 1.0}），样本边际固定 1/B，从 bank 出发；
                  代价矩阵和 getGamma 完全相同。报告的 μ 是计划的行和（总质量 = 1）
                  实现：优先用 ot.unbalanced.sinkhorn_unbalanced(reg_m=(ρ, inf), reg_type="entropy")；
                  POT 版本不支持一侧固定（报错或列边际误差 > 1e-4）时，退回本文件里同公式的半松弛 Sinkhorn-Knopp
                  （u ← (a / Kv)^(ρ/(ρ+ε))，v ← b / Kᵀu），summary.json 里记录实际用了哪种

每个变体报告：计划 argmax 准确率、真类质量、缺席类质量、μ 与本批真实直方图的 L1 距离，
以及 headroom 恢复比例 = (acc_变体 − acc_bank) / (acc_true_batch − acc_bank)（用全部批的均值计算）

用法（在项目根目录下）：
  python tools/diag_mu.py --snapshot process/DiagFix/Ar2Cl/snapshot_p000.pt --out diag_mu_results/Ar2Cl/p000
  python tools/diag_mu.py --collect diag_mu_results        # 汇总所有结果并打包 zip
"""
import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、process/）
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

import numpy as np
import torch
import ot
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from diag_ot import DiagDataset, run_sinkhorn, describe, write_csv
from utils.adapterVIT import clipAdapter
from utils.otUse import getGamma
from model.network import resnet, classifier

SHARP_TS = [0.5, 0.25]
PRESENT_TAUS = [0.3, 0.5]
EM_ITERS = [1, 3]
UOT_RHOS = [0.1, 1.0]
VARIANTS = (["bank", "true_batch", "uniform", "clip_now", "model", "clip_model_avg"]
            + [f"sharp_T{t}" for t in SHARP_TS]
            + [f"present_tau{t}" for t in PRESENT_TAUS]
            + [f"em_rowsum{k}" for k in EM_ITERS]
            + [f"em_mix{k}" for k in EM_ITERS]
            + [f"uot_m{r}" for r in UOT_RHOS])
METRICS = ["plan_acc", "true_mass", "absent_mass", "mu_l1", "mu_eff"]


# =============================================================================
# 小工具
# =============================================================================
def norm_mu(mu):
    """Sinkhorn 需要正的边际：加极小值后重新归一化（同 diag_quick 对 true_batch 的处理）"""
    mu = mu.clamp_min(1e-8)
    return mu / mu.sum()


def ot_cost(clip_prob):
    """和 utils/otUse.getGamma 完全相同的代价矩阵 [C, B]（float64 numpy）"""
    P = clip_prob.T                                         # [C, B]
    feat_dist = 1 - P
    M = (1.0 / feat_dist.max()) * feat_dist - torch.log(P + 1e-6)
    return M.double().cpu().numpy()


def uot_plan(clip_prob, mu0, reg, rho):
    """半松弛 OT：行（类别）边际 KL 松弛 ρ，列（样本）边际固定 1/B。返回 ([C, B] 计划, 用了哪种实现)"""
    M = ot_cost(clip_prob)
    C, B = M.shape
    a = mu0.double().cpu().numpy()
    b = np.full(B, 1.0 / B)
    try:
        g = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, (rho, float("inf")), reg_type="entropy",
                                              numItermax=2000, stopThr=1e-9)
        g = np.asarray(g, dtype=np.float64)
        if np.isfinite(g).all() and np.abs(g.sum(axis=0) - b).sum() < 1e-4:
            return g, "pot"
    except Exception:
        pass
    # 回退：自己实现的半松弛 Sinkhorn-Knopp（熵正则，和 POT reg_type="entropy" 同一公式）
    K = np.exp(-M / reg)
    fi = rho / (rho + reg)
    u, v = np.ones(C), np.ones(B)
    for _ in range(2000):
        u = (a / np.maximum(K @ v, 1e-300)) ** fi
        v = b / np.maximum(K.T @ u, 1e-300)
    return u[:, None] * K * v[None, :], "own"


# =============================================================================
# 诊断
# =============================================================================
def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()

    # ---------------- 读取快照（和 diag_quick.py 相同）----------------
    snap = torch.load(cli.snapshot, map_location="cpu", weights_only=False)
    A = Namespace(**snap["args"])
    A.device = device
    C, B = A.class_num, cli.batch_size
    SCALE = 100.0                                   # main.py 中写死的 CLIP 尺度
    print(f"[快照] {cli.snapshot}  progress={snap['progress']}  iter={snap['iter']}/{snap['max_iter']}")

    raw_clip, preprocess = clip.load(A.clipVersion, device=device)
    clip_model = clipAdapter(raw_clip, device, adapterCount=A.adaptercount)
    missing, unexpected = clip_model.load_state_dict(snap["adapter"], strict=False)
    assert not unexpected and not [k for k in missing if "adapter" in k], "adapter 参数载入不完整"
    clip_model.eval()
    feature_net = resnet(res_name=A.arch).to(device)
    feature_net.load_state_dict(snap["feature"])
    feature_net.train()                             # 和训练时一样用批统计量
    classifier_net = classifier(class_num=C).to(device)
    classifier_net.load_state_dict(snap["classifier"])
    classifier_net.eval()

    vil_bank = snap["vil_bank"].to(device)

    dataset = DiagDataset(snap["imgs"], preprocess, A.class_info, isTraining=False)
    N = len(dataset)

    # ---------------- 预计算当前 CLIP 对全部图片的相似度 ----------------
    sim_all = torch.zeros(N, C, device=device)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        text_feat = clip_model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        for item in tqdm(DataLoader(dataset, batch_size=128, num_workers=cli.num_workers), desc="CLIP 相似度"):
            idx = item["index"].to(device)
            f = clip_model.encode_image(item["clip_img"].to(device).float())
            sim_all[idx] = (f / f.norm(dim=-1, keepdim=True)) @ text_feat.T

    rows = []
    uot_backend = {f"uot_m{r}": {"pot": 0, "own": 0} for r in UOT_RHOS}
    present_fallback = {f"present_tau{t}": 0 for t in PRESENT_TAUS}
    for seed in cli.seeds:
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
        batches = perm[: N // B * B].view(-1, B).tolist()             # 和 diag_quick.py 同样的批划分
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=cli.num_workers, pin_memory=True)

        for bi, item in enumerate(tqdm(loader, desc=f"seed {seed}")):
            idx = item["index"].to(device)
            y = item["label"].to(device)
            absent = torch.ones(C, dtype=torch.bool, device=device)
            absent[y] = False
            cols = torch.arange(B, device=device)

            with torch.no_grad():
                clip_prob = torch.softmax(SCALE * sim_all[idx], dim=1)      # [B, C]，同 main.py
                model_prob = torch.softmax(classifier_net(feature_net(item["img"].to(device).float())), dim=1)
                true_hist = torch.bincount(y, minlength=C).float() / B

                row = {"seed": seed, "batch": bi,
                       "clip_acc": (clip_prob.argmax(dim=1) == y).float().mean().item(),
                       "model_acc": (model_prob.argmax(dim=1) == y).float().mean().item(),
                       "n_present": int((~absent).sum().item())}

                # ---------------- 各种 μ_p 估计 ----------------
                bank = vil_bank[idx].mean(dim=0)
                clip_now = clip_prob.mean(dim=0)
                model_mu = model_prob.mean(dim=0)
                mus = {
                    "bank": bank,
                    "true_batch": true_hist,
                    "uniform": torch.full((C,), 1.0 / C, device=device),
                    "clip_now": clip_now,
                    "model": model_mu,
                    "clip_model_avg": 0.5 * (clip_now + model_mu),
                }
                for t in SHARP_TS:
                    q = clip_prob ** (1.0 / t)
                    mus[f"sharp_T{t}"] = (q / q.sum(dim=1, keepdim=True)).mean(dim=0)
                for t in PRESENT_TAUS:
                    keep = clip_prob.max(dim=0).values >= t
                    m_ = bank * keep
                    if m_.sum() <= 0:
                        m_ = bank
                        present_fallback[f"present_tau{t}"] += 1
                    mus[f"present_tau{t}"] = m_ / m_.sum()
                # 混合比例 EM（CLIP 概率当似然）
                m_ = norm_mu(bank)
                for k in range(1, max(EM_ITERS) + 1):
                    r = clip_prob * m_.unsqueeze(0)
                    m_ = (r / r.sum(dim=1, keepdim=True)).mean(dim=0)
                    if k in EM_ITERS:
                        mus[f"em_mix{k}"] = m_

                # ---------------- 每个变体算计划 ----------------
                plans = {}
                for name in VARIANTS:
                    if name.startswith("em_rowsum") or name.startswith("uot_m"):
                        continue
                    mu = norm_mu(mus[name])
                    mus[name] = mu
                    plans[name], _ = run_sinkhorn(getGamma, A, clip_prob.T, mu, A.otReg)   # [C, B]
                # em_rowsum：μ ← 计划行和（平衡 Sinkhorn 下应 ≈ bank，健全性检查）
                g = plans["bank"]
                for k in range(1, max(EM_ITERS) + 1):
                    mu = norm_mu(g.sum(dim=1)) if g.sum() > 0 and not torch.isnan(g).any() else mus["bank"]
                    g, _ = run_sinkhorn(getGamma, A, clip_prob.T, mu, A.otReg)
                    if k in EM_ITERS:
                        mus[f"em_rowsum{k}"], plans[f"em_rowsum{k}"] = mu, g
                # 半松弛 OT
                for rho in UOT_RHOS:
                    name = f"uot_m{rho}"
                    gn, backend = uot_plan(clip_prob, mus["bank"], A.otReg, rho)
                    uot_backend[name][backend] += 1
                    g = torch.tensor(gn, device=device, dtype=torch.float32)
                    plans[name] = g
                    mus[name] = g.sum(dim=1) / g.sum().clamp_min(1e-30)

                # ---------------- 指标 ----------------
                for name in VARIANTS:
                    g, mu = plans[name], mus[name]
                    row[f"{name}_mu_l1"] = (mu - true_hist).abs().sum().item()
                    row[f"{name}_mu_eff"] = torch.exp(-(mu * torch.log(mu.clamp_min(1e-30))).sum()).item()   # μ 的有效类数
                    if torch.isnan(g).any() or g.sum() <= 0:
                        for k in ("plan_acc", "true_mass", "absent_mass"):
                            row[f"{name}_{k}"] = float("nan")
                        continue
                    row[f"{name}_plan_acc"] = (g.argmax(0) == y).float().mean().item()
                    row[f"{name}_true_mass"] = (g[y, cols].sum() / g.sum()).item()          # 落在真类上的质量
                    row[f"{name}_absent_mass"] = (g[absent].sum() / g.sum()).item()         # 落在缺席类上的质量
                rows.append(row)

    write_csv(os.path.join(cli.out, "batches.csv"), rows)

    # ---------------- 汇总 ----------------
    stats = {k: describe([r[k] for r in rows]) for k in rows[0] if k not in ("seed", "batch")}
    m = lambda k: stats[k].get("mean")
    acc_bank, acc_true = m("bank_plan_acc"), m("true_batch_plan_acc")
    gap = acc_true - acc_bank if acc_true is not None and acc_bank is not None else None
    variants = {}
    for n in VARIANTS:
        d = {k: m(f"{n}_{k}") for k in METRICS}
        d["headroom"] = (d["plan_acc"] - acc_bank) / gap if gap and d["plan_acc"] is not None and abs(gap) > 1e-9 else None
        variants[n] = d
    summary = {
        "snapshot": cli.snapshot, "progress": snap["progress"], "task": f"{A.source}->{A.target}",
        "batch_size": B, "seeds": cli.seeds, "n_batches": len(rows), "otReg": A.otReg,
        "clip_acc": m("clip_acc"), "model_acc": m("model_acc"), "n_present": m("n_present"),
        "headroom_gap": gap, "variants": variants,
        "uot_backend": uot_backend, "present_fallback": present_fallback,
        "stats": stats,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(cli.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 打印 ----------------
    f4 = lambda v: "     -" if v is None else f"{v:.4f}"
    print("\n" + "=" * 78)
    print(f"{summary['task']}  progress={summary['progress']}  批数={len(rows)}  "
          f"CLIP准确率={m('clip_acc'):.4f}  模型准确率={m('model_acc'):.4f}  批内类数={m('n_present'):.1f}")
    print(f"  bank → true_batch 差距 = {f4(gap)}")
    print("  变体              计划准确率  真类质量  缺席质量  μ L1     有效类数  恢复比例")
    for n in VARIANTS:
        d = variants[n]
        print(f"  {n:16s}  {f4(d['plan_acc'])}    {f4(d['true_mass'])}   {f4(d['absent_mass'])}   "
              f"{f4(d['mu_l1'])}  {f4(d['mu_eff'])}   {f4(d['headroom'])}")
    print(f"  半松弛 OT 实现：{uot_backend}   present 退回 bank 次数：{present_fallback}")
    print("=" * 78)
    print(f"结果已保存到 {cli.out}")


# =============================================================================
# 汇总所有结果并打包
# =============================================================================
def collect(root):
    files = sorted(glob.glob(os.path.join(root, "*", "*", "summary.json")))
    if not files:
        print(f"[错误] {root} 下没有找到 summary.json"); return
    S = [json.load(open(f, encoding="utf-8")) for f in files]
    f4 = lambda v: "-" if v is None else f"{v:.4f}"
    md = ["## μ_p 估计器对比（每行一个 任务 / 进度 / 变体）", "",
          "恢复比例 = (acc_变体 − acc_bank) / (acc_true_batch − acc_bank)", "",
          "| 任务 | 进度 | CLIP | 变体 | 计划准确率 | 真类质量 | 缺席质量 | μ L1 | 有效类数 | 恢复比例 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for n in VARIANTS:
            d = s["variants"].get(n)
            if d is None:
                continue
            md.append(f"| {s['task']} | {s['progress']} | {f4(s['clip_acc'])} | {n} | {f4(d['plan_acc'])} | "
                      f"{f4(d['true_mass'])} | {f4(d['absent_mass'])} | {f4(d['mu_l1'])} | {f4(d['mu_eff'])} | "
                      f"{f4(d['headroom'])} |")
    md += ["", "## 恢复比例速览", "",
           "| 任务 | 进度 | 差距 | " + " | ".join(VARIANTS) + " |", "|" + "---|" * (3 + len(VARIANTS))]
    for s in S:
        md.append(f"| {s['task']} | {s['progress']} | {f4(s['headroom_gap'])} | " +
                  " | ".join(f4(s["variants"].get(n, {}).get("headroom")) for n in VARIANTS) + " |")
    md += ["", "## 实现记录", "", "| 任务 | 进度 | 半松弛 OT 实现 | present 退回 bank 次数 |", "|---|---|---|---|"]
    for s in S:
        md.append(f"| {s['task']} | {s['progress']} | {json.dumps(s['uot_backend'])} | {json.dumps(s['present_fallback'])} |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(root, "summary_all.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    zip_path = shutil.make_archive(root.rstrip("/"), "zip", root_dir=root)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="μ_p 估计诊断：不用标签的类别边际估计器能收回多少 oracle 差距")
    parser.add_argument("--snapshot", type=str, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, help="结果输出目录")
    parser.add_argument("--collect", type=str, default=None, help="只做汇总打包：结果根目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli.collect)
    else:
        assert cli.snapshot and cli.out, "需要 --snapshot 和 --out"
        run(cli)
