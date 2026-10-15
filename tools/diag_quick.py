"""
快速诊断（不训练，直接用 scripts/run_diag.sh 保存的快照）

  [2] μ_p 消融：同一批、同一 CLIP 概率，只换 Sinkhorn 的类别边际 μ_p，看计划质量
        bank        记忆库 CLIP 概率的批均值（main.py 实际用的）
        clip_now    当前 CLIP 概率的批均值（记忆库不过期时的样子）
        uniform     均匀分布
        true_global 目标域真实类频率（上界参考，实际拿不到）
        true_batch  本批真实类频率（oracle，缺席类质量必为 0）
  [3] 缺席类质量落在哪：CLIP 判错的样本 vs 判对的样本；计划把 CLIP 的错纠正了多少 / 改坏了多少
  [H3 修正] 分母改用 gamma1 实际的行 / 列边际求精确最优，使分子分母在同一可行集里比较

用法（在项目根目录下）：
  python tools/diag_quick.py --snapshot process/Diag/Ar2Cl/snapshot_p000.pt --out diag_quick_results/Ar2Cl/p000
  python tools/diag_quick.py --collect diag_quick_results        # 汇总所有结果并打包 zip
"""
import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、process/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
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
from utils.otUse import getGamma, getGamma_self
from model.network import resnet, classifier

MU_NAMES = ["bank", "clip_now", "uniform", "true_global", "true_batch"]


# =============================================================================
# 诊断
# =============================================================================
def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()

    # ---------------- 读取快照（和 diag_ot.py 相同）----------------
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
    feature_net.train()                             # 和训练时计算 Cm 一样用批统计量
    classifier_net = classifier(class_num=C).to(device)
    classifier_net.load_state_dict(snap["classifier"])
    classifier_net.eval()

    vil_bank = snap["vil_bank"].to(device)
    model_bank = snap["model_bank"].to(device)
    class_w = 1.0 / torch.clamp(model_bank.mean(dim=0), min=1e-4)     # main.py 的 1/类频率重加权

    dataset = DiagDataset(snap["imgs"], preprocess, A.class_info, isTraining=False)
    N = len(dataset)

    # ---------------- 预计算当前 CLIP 对全部图片的相似度 ----------------
    sim_all = torch.zeros(N, C, device=device)
    labels_all = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        text_feat = clip_model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        for item in tqdm(DataLoader(dataset, batch_size=128, num_workers=cli.num_workers), desc="CLIP 相似度"):
            idx = item["index"].to(device)
            f = clip_model.encode_image(item["clip_img"].to(device).float())
            sim_all[idx] = (f / f.norm(dim=-1, keepdim=True)) @ text_feat.T
            labels_all[idx] = item["label"].to(device)
    true_freq = torch.bincount(labels_all, minlength=C).float() / N   # 目标域真实类频率

    rows = []
    for seed in cli.seeds:
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
        batches = perm[: N // B * B].view(-1, B).tolist()             # 和 diag_ot.py 同样的批划分
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=cli.num_workers, pin_memory=True)

        for bi, item in enumerate(tqdm(loader, desc=f"seed {seed}")):
            idx = item["index"].to(device)
            y = item["label"].to(device)
            absent = torch.ones(C, dtype=torch.bool, device=device)
            absent[y] = False
            cols = torch.arange(B, device=device)

            with torch.no_grad():
                clip_prob = torch.softmax(SCALE * sim_all[idx], dim=1)      # [B, C]，同 main.py
                clip_pred = clip_prob.argmax(dim=1)
                clip_ok = clip_pred == y
                feat = feature_net(item["img"].to(device).float())
                model_prob = torch.softmax(classifier_net(feat), dim=1)

                row = {"seed": seed, "batch": bi,
                       "clip_acc": clip_ok.float().mean().item(),
                       "clip_wrong_frac": (~clip_ok).float().mean().item()}

                mus = {
                    "bank": vil_bank[idx].mean(dim=0),
                    "clip_now": clip_prob.mean(dim=0),
                    "uniform": torch.full((C,), 1.0 / C, device=device),
                    "true_global": true_freq,
                    "true_batch": torch.bincount(y, minlength=C).float() / B,
                }

                # ---------------- [2] μ_p 消融 ----------------
                g_bank = None
                for name in MU_NAMES:
                    # true_batch 在缺席类上为 0，Sinkhorn 需要正的边际，加极小值后重新归一化
                    mu = mus[name].clamp_min(1e-8)
                    mu = mu / mu.sum()
                    g, _ = run_sinkhorn(getGamma, A, clip_prob.T, mu, A.otReg)   # [C, B]
                    if name == "bank":
                        g_bank = g
                    if torch.isnan(g).any() or g.sum() <= 0:
                        for k in ("plan_acc", "true_mass", "absent_mass", "plan_acc_rw", "true_mass_rw"):
                            row[f"{name}_{k}"] = float("nan")
                        continue
                    gw = g * class_w.unsqueeze(1)
                    row[f"{name}_plan_acc"] = (g.argmax(0) == y).float().mean().item()
                    row[f"{name}_true_mass"] = (g[y, cols].sum() / g.sum()).item()          # 落在真类上的质量
                    row[f"{name}_absent_mass"] = (g[absent].sum() / g.sum()).item()
                    row[f"{name}_plan_acc_rw"] = (gw.argmax(0) == y).float().mean().item()  # 重加权后（进入损失的计划）
                    row[f"{name}_true_mass_rw"] = (gw[y, cols].sum() / gw.sum()).item()

                # ---------------- [3] 缺席类质量落在哪（用训练实际的 bank μ_p）----------------
                g = g_bank
                col_mass = g.sum(dim=0)                                    # 每个样本拿到的总质量（≈1/B）
                col_absent = g[absent].sum(dim=0) / col_mass               # 每个样本质量中落在缺席类的比例
                plan_ok = g.argmax(0) == y
                n_wrong = int((~clip_ok).sum().item())
                n_ok = B - n_wrong
                row["absent_share_on_clip_wrong"] = (                      # 缺席质量中落在 CLIP 判错样本上的份额
                    (col_absent * col_mass)[~clip_ok].sum() / (col_absent * col_mass).sum()).item()
                row["absent_frac_per_wrong"] = col_absent[~clip_ok].mean().item() if n_wrong else float("nan")
                row["absent_frac_per_ok"] = col_absent[clip_ok].mean().item() if n_ok else float("nan")
                # CLIP 判错时，它预测的类是否本来就不在这一批里
                row["wrong_pred_is_absent"] = absent[clip_pred[~clip_ok]].float().mean().item() if n_wrong else float("nan")
                # 计划相对 CLIP argmax：纠正了多少错、改坏了多少对
                row["plan_rescue_rate"] = (plan_ok & ~clip_ok).sum().item() / n_wrong if n_wrong else float("nan")
                row["plan_break_rate"] = (~plan_ok & clip_ok).sum().item() / n_ok if n_ok else float("nan")
                row["plan_rescue_n"] = int((plan_ok & ~clip_ok).sum().item())
                row["plan_break_n"] = int((~plan_ok & clip_ok).sum().item())

                # ---------------- [H3 修正] 分母用 gamma1 实际的边际 ----------------
                _, Cm = getGamma_self(A, model_prob, feat, model_bank[idx].mean(dim=0), classifier_net, reg=A.otReg)
                Cm = Cm.detach().double().cpu().numpy()
                for tag, gg in [("cv", g), ("cv_rw", g * class_w.unsqueeze(1))]:
                    gn = gg.double().cpu().numpy()
                    gn = gn / gn.sum()
                    a, b = gn.sum(axis=1), gn.sum(axis=0)                  # gamma 实际的行 / 列边际
                    emd = ot.emd(a, b, Cm)
                    opt = float((emd * Cm).sum())
                    row[f"{tag}_cost"] = float((gn * Cm).sum())
                    row[f"{tag}_opt_cost"] = opt
                    row[f"{tag}_ratio_fixed"] = row[f"{tag}_cost"] / opt if opt > 0 else float("nan")
                row["cv_col_err"] = float(np.abs(g.double().cpu().numpy().sum(axis=0) - 1.0 / B).sum())
                rows.append(row)

    write_csv(os.path.join(cli.out, "batches.csv"), rows)

    # ---------------- 汇总 ----------------
    stats = {k: describe([r[k] for r in rows]) for k in rows[0] if k not in ("seed", "batch")}
    m = lambda k: stats[k].get("mean")
    summary = {
        "snapshot": cli.snapshot, "progress": snap["progress"], "task": f"{A.source}->{A.target}",
        "batch_size": B, "seeds": cli.seeds, "n_batches": len(rows), "otReg": A.otReg,
        "stats": stats,
        "mu_ablation": {n: {k: m(f"{n}_{k}") for k in ("plan_acc", "true_mass", "absent_mass", "plan_acc_rw", "true_mass_rw")}
                        for n in MU_NAMES},
        "absent_location": {k: m(k) for k in ("clip_wrong_frac", "absent_share_on_clip_wrong", "absent_frac_per_wrong",
                                              "absent_frac_per_ok", "wrong_pred_is_absent", "plan_rescue_rate", "plan_break_rate")},
        "absent_location_counts": {"rescue_total": int(sum(r["plan_rescue_n"] for r in rows)),
                                   "break_total": int(sum(r["plan_break_n"] for r in rows))},
        "h3_fixed": {"ratio_cv": m("cv_ratio_fixed"), "ratio_cv_median": stats["cv_ratio_fixed"].get("median"),
                     "ratio_cv_min": stats["cv_ratio_fixed"].get("min"),
                     "ratio_cv_rw": m("cv_rw_ratio_fixed"), "col_err": m("cv_col_err")},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(cli.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------- 打印 ----------------
    print("\n" + "=" * 70)
    print(f"{summary['task']}  progress={summary['progress']}  批数={len(rows)}  CLIP准确率={m('clip_acc'):.4f}")
    print("  [2] μ_p 消融       计划准确率  真类质量  缺席质量  | 重加权后: 准确率  真类质量")
    for n in MU_NAMES:
        d = summary["mu_ablation"][n]
        print(f"      {n:12s}  {d['plan_acc']:.4f}    {d['true_mass']:.4f}   {d['absent_mass']:.4f}  |          {d['plan_acc_rw']:.4f}  {d['true_mass_rw']:.4f}")
    al, ct = summary["absent_location"], summary["absent_location_counts"]
    print(f"  [3] CLIP 判错样本占比={al['clip_wrong_frac']:.3f}  缺席质量落在判错样本上的份额={al['absent_share_on_clip_wrong']:.3f}")
    print(f"      每个判错样本的缺席质量比例={al['absent_frac_per_wrong']:.3f}  每个判对样本={al['absent_frac_per_ok']:.3f}")
    print(f"      判错时预测类不在本批的比例={al['wrong_pred_is_absent']:.3f}")
    print(f"      计划纠正 CLIP 错误={al['plan_rescue_rate']:.4f}（共 {ct['rescue_total']}）  改坏 CLIP 正确={al['plan_break_rate']:.4f}（共 {ct['break_total']}）")
    h = summary["h3_fixed"]
    print(f"  [H3 修正] 代价比 Cv={h['ratio_cv']:.4f}（中位 {h['ratio_cv_median']:.4f}，最小 {h['ratio_cv_min']:.4f}）  重加权={h['ratio_cv_rw']:.4f}")
    print("=" * 70)
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
    md = ["## [2] μ_p 消融（计划准确率 / 真类质量 / 缺席质量）", "",
          "| 任务 | 进度 | CLIP | " + " | ".join(MU_NAMES) + " |", "|" + "---|" * (3 + len(MU_NAMES))]
    for s in S:
        cells = [f"{f4(s['mu_ablation'][n]['plan_acc'])} / {f4(s['mu_ablation'][n]['true_mass'])} / {f4(s['mu_ablation'][n]['absent_mass'])}"
                 for n in MU_NAMES]
        md.append(f"| {s['task']} | {s['progress']} | {f4(s['stats']['clip_acc']['mean'])} | " + " | ".join(cells) + " |")
    keys = ["clip_wrong_frac", "absent_share_on_clip_wrong", "absent_frac_per_wrong", "absent_frac_per_ok",
            "wrong_pred_is_absent", "plan_rescue_rate", "plan_break_rate"]
    md += ["", "## [3] 缺席类质量落在哪", "", "| 任务 | 进度 | " + " | ".join(keys) + " | 纠正数 | 改坏数 |", "|" + "---|" * (4 + len(keys))]
    for s in S:
        al, ct = s["absent_location"], s["absent_location_counts"]
        md.append(f"| {s['task']} | {s['progress']} | " + " | ".join(f4(al[k]) for k in keys) +
                  f" | {ct['rescue_total']} | {ct['break_total']} |")
    md += ["", "## [H3 修正] 代价比", "", "| 任务 | 进度 | Cv 均值 | Cv 中位 | Cv 最小 | Cv 重加权 | 列边际误差 |", "|---|---|---|---|---|---|---|"]
    for s in S:
        h = s["h3_fixed"]
        md.append(f"| {s['task']} | {s['progress']} | {f4(h['ratio_cv'])} | {f4(h['ratio_cv_median'])} | {f4(h['ratio_cv_min'])} | "
                  f"{f4(h['ratio_cv_rw'])} | {h['col_err']:.2e} |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(root, "summary_all.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    zip_path = shutil.make_archive(root.rstrip("/"), "zip", root_dir=root)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="快速诊断：μ_p 消融 / 缺席质量位置 / H3 修正")
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
