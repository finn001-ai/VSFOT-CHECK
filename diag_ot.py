"""
实验 0：OT 静态诊断（不训练）

读取 main.py 保存的快照（--snapshot_at），对目标域逐批计算：
  A. 批内类覆盖 / 缺席类质量：批内出现几类，Sinkhorn 计划把多少质量分给了批内不存在的类
  B. 尺度敏感：改 CLIP logit 尺度和 ε，看计划的熵、计划 argmax 与真标签的一致度
  C. 理论缺口：Cm 下的精确最优代价 vs Cv 方案（训练实际用的 gamma1）在 Cm 下的代价

用法（在项目根目录下）：
  python diag_ot.py --snapshot process/Diag/Ar2Cl/snapshot_p000.pt --out diag_results/Ar2Cl/p000

输出（--out 目录下）：
  batches.csv    每批一行，A 和 C 的逐批指标
  grid.csv       每批 × 每个 (模式, 尺度, ε) 一行，B 的逐批指标
  summary.json   汇总统计（均值 / 中位数 / P90）和三个判据的结论
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import csv
import json
import time
import argparse
import warnings
from argparse import Namespace

import numpy as np
import torch
import ot
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.Dataset import dataset5X, RGB_loader
from utils.adapterVIT import clipAdapter
from utils.otUse import getGamma, getGamma_self
from model.network import resnet, classifier


# =============================================================================
# 数据：只取诊断需要的字段，跳过训练用的强增广，节省 CPU
# =============================================================================
class DiagDataset(dataset5X):
    def __getitem__(self, index):
        imgs_path, imgs_label = self.imgs[index]
        img = RGB_loader(imgs_path)
        return {
            "img": self.test_transform(img),        # 模型输入（确定性的测试变换）
            "clip_img": self.clipProcessVal(img),   # CLIP 输入
            "label": int(imgs_label),
            "index": int(index),                    # 记忆库索引
        }


# =============================================================================
# 小工具
# =============================================================================
def run_sinkhorn(fn, *a, **kw):
    """调用项目里的 getGamma / getGamma_self，同时记录 Sinkhorn 是否报未收敛警告"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    not_conv = any("converge" in str(x.message).lower() for x in w)
    return out, not_conv


def plan_stats(g, y, absent, w=None):
    """
    计划 g: [C, B]，y: [B] 真标签，absent: [C] bool 缺席类
    w: 类权重（main.py 中 Sinkhorn 之后乘上的 1/类频率），为 None 则不重加权
    """
    # ε 太小时 float32 下 exp(-cost/ε) 会下溢成 0，Sinkhorn 得到 NaN；此时指标记为 NaN，不参与统计
    if torch.isnan(g).any() or g.sum() <= 0:
        nan = float("nan")
        return {"absent_mass": nan, "plan_acc": nan, "col_entropy": nan, "eff_classes": nan, "has_nan": True}
    if w is not None:
        g = g * w.unsqueeze(1)
    total = g.sum()
    col = g / g.sum(dim=0, keepdim=True).clamp_min(1e-30)              # 每个样本在各类上的分布
    ent = -(col * torch.log(col.clamp_min(1e-30))).sum(dim=0)          # 每列的熵（nats）
    return {
        "absent_mass": (g[absent].sum() / total).item(),               # 缺席类拿到的质量占比
        "plan_acc": (g.argmax(dim=0) == y).float().mean().item(),      # 计划 argmax 与真标签一致度
        "col_entropy": ent.mean().item(),                              # 平均列熵
        "eff_classes": torch.exp(ent).mean().item(),                   # 平均「有效类数」= exp(熵)
        "has_nan": False,
    }


def marginal_err(g, mu, B):
    """Sinkhorn 实际满足边际的程度（L1 误差），用来判断是否收敛"""
    row = (g.sum(dim=1) - mu).abs().sum().item()
    col = (g.sum(dim=0) - 1.0 / B).abs().sum().item()
    return row, col


def describe(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="OT 静态诊断")
    parser.add_argument("--snapshot", type=str, required=True, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, required=True, help="结果输出目录")
    parser.add_argument("--device", type=int, default=0, help="GPU 编号；-1 表示用 CPU（仅用于调试）")
    parser.add_argument("--batch_size", type=int, default=64, help="诊断用的批大小")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="每个种子随机打乱一遍数据，遍历一整个 epoch")
    parser.add_argument("--scales", type=float, nargs="+", default=[50, 100, 200], help="CLIP logit 尺度网格（代码默认 100）")
    parser.add_argument("--regs", type=float, nargs="+", default=[0.1, 0.2, 0.4], help="Sinkhorn ε 网格（代码默认 0.2）")
    parser.add_argument("--bn_mode", type=str, default="train", choices=["train", "eval"],
                        help="模型 BN 模式；训练时计算 Cm 用的是 train 模式（批统计量）")
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}" if cli.device >= 0 else "cpu"
    t0 = time.time()

    # ---------------- 读取快照，恢复训练时的配置 ----------------
    snap = torch.load(cli.snapshot, map_location="cpu", weights_only=False)
    A = Namespace(**snap["args"])          # 训练时的 args（含 class_info、otReg、adaptercount 等）
    A.device = device
    C = A.class_num
    B = cli.batch_size
    BASE_SCALE = 100.0                     # main.py 中写死的 CLIP 尺度
    print(f"[快照] {cli.snapshot}  progress={snap['progress']}  iter={snap['iter']}/{snap['max_iter']}")
    print(f"[任务] {A.source} -> {A.target}  C={C}  B={B}  otReg={A.otReg}  bn_mode={cli.bn_mode}")

    # ---------------- 恢复模型 ----------------
    raw_clip, preprocess = clip.load(A.clipVersion, device=device)
    clip_model = clipAdapter(raw_clip, device, adapterCount=A.adaptercount)
    missing, unexpected = clip_model.load_state_dict(snap["adapter"], strict=False)
    assert not unexpected, f"adapter 参数多余: {unexpected[:3]}"
    assert not [k for k in missing if "adapter" in k], "adapter 参数没有完整载入"
    clip_model.eval()

    feature_net = resnet(res_name=A.arch).to(device)
    feature_net.load_state_dict(snap["feature"])
    classifier_net = classifier(class_num=C).to(device)
    classifier_net.load_state_dict(snap["classifier"])
    feature_net.train() if cli.bn_mode == "train" else feature_net.eval()
    classifier_net.eval()

    vil_bank = snap["vil_bank"].to(device)        # [N, C] 训练时记忆库里的 CLIP 概率 -> getGamma 的 μ_p
    model_bank = snap["model_bank"].to(device)    # [N, C] 训练时记忆库里的模型概率
    # main.py：gamma1 在 Sinkhorn 之后乘以 1/类频率
    class_w = 1.0 / torch.clamp(model_bank.mean(dim=0), min=1e-4)

    # 图片顺序必须和记忆库索引一致，所以用快照里保存的列表
    dataset = DiagDataset(snap["imgs"], preprocess, A.class_info, isTraining=False)
    N = len(dataset)
    assert N == vil_bank.shape[0], "图片数量和记忆库大小不一致"

    # ---------------- 预计算：当前 CLIP 对全部图片的相似度 ----------------
    # 用于「尺度」实验中按新尺度重算 μ_p（main.py 的 μ_p 来自记忆库的 CLIP 概率）
    sim_all = torch.zeros(N, C, device=device)
    labels_all = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        text_feat = clip_model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=cli.num_workers)
        for item in tqdm(loader, desc="CLIP 相似度"):
            idx = item["index"].to(device)
            f = clip_model.encode_image(item["clip_img"].to(device).float())
            f = f / f.norm(dim=-1, keepdim=True)
            sim_all[idx] = f @ text_feat.T
            labels_all[idx] = item["label"].to(device)
    clip_zs_acc = (sim_all.argmax(1) == labels_all).float().mean().item()
    print(f"[参考] 当前 CLIP 全量零样本准确率: {clip_zs_acc:.4f}")

    # ---------------- 逐批诊断 ----------------
    batch_rows, grid_rows = [], []
    for seed in cli.seeds:
        g_ = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g_)
        batches = perm[: N // B * B].view(-1, B).tolist()     # 丢弃最后不满一批的（和训练 drop_last 一致）
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=cli.num_workers, pin_memory=True)

        for bi, item in enumerate(tqdm(loader, desc=f"seed {seed}")):
            idx = item["index"].to(device)
            y = item["label"].to(device)
            present = torch.zeros(C, dtype=torch.bool, device=device)
            present[y] = True
            absent = ~present

            with torch.no_grad():
                feat = feature_net(item["img"].to(device).float())
                model_prob = torch.softmax(classifier_net(feat), dim=1)       # [B, C]
                sim = sim_all[idx]                                            # [B, C]
                clip_prob = torch.softmax(BASE_SCALE * sim, dim=1)            # [B, C]，同 main.py
                mu_bank = vil_bank[idx].mean(dim=0)                           # 同 MB.get -> vilMemoryMean
                mu_model = model_bank[idx].mean(dim=0)                        # 同 MB.get -> modelMemoryMean

                row = {
                    "seed": seed, "batch": bi,
                    "n_present": int(present.sum().item()),
                    "clip_acc": (clip_prob.argmax(1) == y).float().mean().item(),
                    "model_acc": (model_prob.argmax(1) == y).float().mean().item(),
                    "mu_bank_absent": mu_bank[absent].sum().item(),           # μ_p 本身在缺席类上的质量
                    "mu_uniform_absent": absent.sum().item() / C,
                }

                # ---- A. 缺席类质量：均匀 μ_p vs 实际 μ_p（记忆库）；重加权前 / 后 ----
                gammas = {}
                for mu_name, mu_arg, mu_vec in [("uniform", None, torch.full((C,), 1.0 / C, device=device)),
                                                ("bank", mu_bank, mu_bank)]:
                    g, nc = run_sinkhorn(getGamma, A, clip_prob.T, mu_arg, A.otReg)
                    gammas[mu_name] = g
                    re, ce = marginal_err(g, mu_vec, B)
                    row[f"{mu_name}_row_err"] = re
                    row[f"{mu_name}_col_err"] = ce
                    row[f"{mu_name}_not_converged"] = int(nc)
                    for tag, w in [("pre", None), ("post", class_w)]:
                        for k, v in plan_stats(g, y, absent, w).items():
                            row[f"{mu_name}_{tag}_{k}"] = v

                # ---- C. 理论缺口：统一边际（μ_p = 记忆库 CLIP 均值，μ_t = 均匀）----
                _, Cm = getGamma_self(A, model_prob, feat, mu_model, classifier_net, reg=A.otReg)
                Cm = Cm.detach().double().cpu().numpy()                        # [C, B] 模型代价
                mu = mu_bank.double().cpu().numpy(); mu = mu / mu.sum()
                nu = np.full(B, 1.0 / B)
                emd = ot.emd(mu, nu, Cm)                                       # Cm 下的精确最优
                opt = float((emd * Cm).sum())

                def cost_under_Cm(g):
                    g = g.double().cpu().numpy()
                    return float((g / g.sum() * Cm).sum())                     # 归一化后在 Cm 下的代价

                g1 = gammas["bank"]                                            # 即训练中的 gamma1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gm_ent = torch.from_numpy(ot.sinkhorn(mu, nu, Cm, reg=A.otReg))
                row["cm_opt_cost"] = opt
                row["cv_cost"] = cost_under_Cm(g1)
                row["cv_rw_cost"] = cost_under_Cm(g1 * class_w.unsqueeze(1))
                row["cm_entropic_cost"] = cost_under_Cm(gm_ent)
                row["ratio_cv"] = row["cv_cost"] / opt if opt > 0 else float("nan")
                row["ratio_cv_rw"] = row["cv_rw_cost"] / opt if opt > 0 else float("nan")
                row["ratio_cm_entropic"] = row["cm_entropic_cost"] / opt if opt > 0 else float("nan")
                batch_rows.append(row)

                # ---- B. 尺度敏感：尺度 × ε 网格 ----
                # joint：clipProb 和 μ_p 同时按新尺度计算（对应真实训练流程整体换尺度）
                # prob_only：只改 clipProb，μ_p 仍用记忆库（尺度 100）
                for s in cli.scales:
                    cp = torch.softmax(s * sim, dim=1)
                    mu_joint = torch.softmax(s * sim_all[idx], dim=1).mean(dim=0)
                    for mode, mu_vec in [("joint", mu_joint), ("prob_only", mu_bank)]:
                        for eps in cli.regs:
                            g, nc = run_sinkhorn(getGamma, A, cp.T, mu_vec, eps)
                            st = plan_stats(g, y, absent)
                            re, _ = marginal_err(g, mu_vec, B)
                            grid_rows.append({"seed": seed, "batch": bi, "mode": mode, "scale": s, "eps": eps,
                                              "clip_acc": (cp.argmax(1) == y).float().mean().item(),
                                              **st, "row_err": re, "not_converged": int(nc)})

    write_csv(os.path.join(cli.out, "batches.csv"), batch_rows)
    write_csv(os.path.join(cli.out, "grid.csv"), grid_rows)

    # =========================================================================
    # 汇总与判据
    # =========================================================================
    col = lambda rows, k: [r[k] for r in rows]
    summary = {
        "snapshot": cli.snapshot, "progress": snap["progress"], "iter": snap["iter"], "max_iter": snap["max_iter"],
        "task": f"{A.source}->{A.target}", "batch_size": B, "seeds": cli.seeds, "n_batches": len(batch_rows),
        "otReg": A.otReg, "bn_mode": cli.bn_mode, "clip_zeroshot_acc_full": clip_zs_acc,
        "versions": {"torch": torch.__version__, "pot": ot.__version__,
                     "gpu": torch.cuda.get_device_name(device) if cli.device >= 0 else "cpu"},
        "stats": {k: describe(col(batch_rows, k)) for k in batch_rows[0] if k not in ("seed", "batch")},
    }

    # 尺度网格：每个 (mode, scale, eps) 在所有批上的均值
    grid_mean = {}
    for mode in ("joint", "prob_only"):
        for s in cli.scales:
            for eps in cli.regs:
                rs = [r for r in grid_rows if r["mode"] == mode and r["scale"] == s and r["eps"] == eps]
                grid_mean[f"{mode}|{s:g}|{eps:g}"] = {k: float(np.nanmean(col(rs, k)))
                                                     for k in ("plan_acc", "clip_acc", "col_entropy", "eff_classes",
                                                               "absent_mass", "row_err", "not_converged", "has_nan")}
    summary["grid_mean"] = grid_mean

    # 「翻倍范围」内一致度的最大变化（百分点）：相邻的尺度对 / ε 对必须恰好是 2 倍关系
    def doubling_pairs(vals):
        vals = sorted(vals)
        return [(a, b) for a, b in zip(vals, vals[1:]) if abs(b / a - 2) < 1e-6]
    h2 = {}
    for mode in ("joint", "prob_only"):
        d_scale = [abs(grid_mean[f"{mode}|{b:g}|{A.otReg:g}"]["plan_acc"] - grid_mean[f"{mode}|{a:g}|{A.otReg:g}"]["plan_acc"]) * 100
                   for a, b in doubling_pairs(cli.scales) if f"{mode}|{a:g}|{A.otReg:g}" in grid_mean]
        d_eps = [abs(grid_mean[f"{mode}|{BASE_SCALE:g}|{b:g}"]["plan_acc"] - grid_mean[f"{mode}|{BASE_SCALE:g}|{a:g}"]["plan_acc"]) * 100
                 for a, b in doubling_pairs(cli.regs) if f"{mode}|{BASE_SCALE:g}|{a:g}" in grid_mean]
        h2[mode] = {"max_delta_pp_scale": max(d_scale) if d_scale else None,
                    "max_delta_pp_eps": max(d_eps) if d_eps else None}

    st = summary["stats"]
    weak = lambda v, th: None if v is None else bool(v < th)
    summary["criteria"] = {
        "H1_absent_mass": {
            "rule": "缺席类质量 < 5% => 假设 1 弱（取逐批均值）",
            "bank_pre": st["bank_pre_absent_mass"]["mean"],
            "bank_post(进入损失的计划)": st["bank_post_absent_mass"]["mean"],
            "uniform_pre": st["uniform_pre_absent_mass"]["mean"],
            "weak_bank_pre": weak(st["bank_pre_absent_mass"]["mean"], 0.05),
            "weak_bank_post": weak(st["bank_post_absent_mass"]["mean"], 0.05),
        },
        "H2_scale": {
            "rule": "尺度 / ε 翻倍范围内一致度变化 < 1 个百分点 => 假设 2 弱",
            **{f"{m}_{k}": v for m, d in h2.items() for k, v in d.items()},
            "weak_joint": weak(max([v for v in h2["joint"].values() if v is not None], default=None), 1.0),
        },
        "H3_gap": {
            "rule": "Cv 方案在 Cm 下的代价 / Cm 精确最优代价 < 1.1 => 理论缺口无实质（取逐批均值）",
            "ratio_cv": st["ratio_cv"]["mean"],
            "ratio_cv_rw(重加权后)": st["ratio_cv_rw"]["mean"],
            "ratio_cm_entropic(参考：Cm 自身熵正则解)": st["ratio_cm_entropic"]["mean"],
            "weak_cv": weak(st["ratio_cv"]["mean"], 1.1),
            "weak_cv_rw": weak(st["ratio_cv_rw"]["mean"], 1.1),
        },
        "sanity": {
            "bank_not_converged_rate": st["bank_not_converged"]["mean"],
            "bank_row_err_mean": st["bank_row_err"]["mean"],
            "plan_acc_bank_pre": st["bank_pre_plan_acc"]["mean"],
            "clip_acc_batch": st["clip_acc"]["mean"],
            "model_acc_batch": st["model_acc"]["mean"],
        },
    }
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(cli.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 打印简表，方便在 Jupyter 里直接看 ----------------
    cr = summary["criteria"]
    print("\n" + "=" * 70)
    print(f"{summary['task']}  progress={summary['progress']}  批数={summary['n_batches']}  用时={summary['elapsed_sec']}s")
    print(f"  批内平均类数: {st['n_present']['mean']:.1f}（中位 {st['n_present']['median']:.0f}）/ {C}")
    print(f"  准确率  CLIP={cr['sanity']['clip_acc_batch']:.4f}  模型={cr['sanity']['model_acc_batch']:.4f}  "
          f"计划argmax={cr['sanity']['plan_acc_bank_pre']:.4f}")
    # POT 的默认停止阈值 1e-9 在 float32 下基本达不到，所以「未收敛警告」几乎总是出现；是否真正收敛以边际误差为准
    print(f"  Sinkhorn 行边际L1误差={cr['sanity']['bank_row_err_mean']:.2e}（以此判断收敛）  POT未收敛警告比例={cr['sanity']['bank_not_converged_rate']:.2f}")
    h1 = cr["H1_absent_mass"]
    print(f"  [H1] 缺席质量  均匀μ={h1['uniform_pre']:.3f}  实际μ={h1['bank_pre']:.3f}  重加权后={h1['bank_post(进入损失的计划)']:.3f}"
          f"  -> 弱? 前:{h1['weak_bank_pre']} 后:{h1['weak_bank_post']}")
    print(f"  [H2] 翻倍范围最大变化(pp)  joint 尺度={h2['joint']['max_delta_pp_scale']}  ε={h2['joint']['max_delta_pp_eps']}"
          f"  -> 弱? {cr['H2_scale']['weak_joint']}")
    h3 = cr["H3_gap"]
    print(f"  [H3] 代价比  Cv={h3['ratio_cv']:.3f}  Cv重加权={h3['ratio_cv_rw(重加权后)']:.3f}  "
          f"Cm熵解={h3['ratio_cm_entropic(参考：Cm 自身熵正则解)']:.3f}  -> 弱? {h3['weak_cv']}")
    print("=" * 70)
    print(f"结果已保存到 {cli.out}")


if __name__ == "__main__":
    main()
