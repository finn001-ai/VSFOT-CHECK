"""
类原型修正（离线，不训练）：老师的分类头是固定的文本原型，不用标签地移动它能拿回多少（读取 main.py --snapshot_at 保存的快照）

H1（archive/2026_10_20_h1）：学生表示不是瓶颈；学生「探针 − 实际」与老师「探针 − 实际」几乎一一对应（Ar→Pr 3.6 / 3.9），
老师的分类器是固定的文本嵌入（"a photo of a {类名}"），MGVA 只调图像侧 adapter 去贴合它。原始 CLIP 探针 − 零样本：
Clipart 17.8、Product 8.3、Art 5.2、Real World 2.9。这里在冻结的 CLIP 图像特征上，从文本原型出发做无标签的原型修正：
  每轮：用当前原型给全部样本分配（分配方式见下），原型 ← 归一化((1 − λ) · 分配加权的图像特征均值 + λ · 文本原型)
  分配方式  soft    softmax(100 · cos)（和老师的预测同一个温度）
            hard    argmax（k-means）
            sk_uni  Sinkhorn，类边际均匀
            sk_sqrt Sinkhorn，类边际 ∝ √(零样本预测计数)（均衡推得轻一半，见 archive/2026_10_20_teacher）
  λ = 0（只用图像均值）/ 0.5（保留文本先验）；center = 图像特征先减去全域均值再归一化（只用于 λ = 0，文本原型不在同一空间）
评估 = 原型上的 argmax（可以直接作为训练中老师的分类头）。报告第 1 / 3 / 10 轮，不按标签挑轮数；主表用第 10 轮。
另外在学生瓶颈特征上做同样的修正（初始分配 = 学生概率），和之前「用自身预测重拟合线性头只有 ±0.6」（archive/2026_10_18_repr）对照。

用法（一般由 scripts/run_proto.sh 调用）：
  python tools/diag_proto.py --snapshot runs/H1Probe/p000/Ar2Pr_s42/snapshot_p000.pt --out runs/proto_results/proto/Ar2Pr/p000
  python tools/diag_proto.py --collect runs/proto_results/proto --h1_root runs/h1_results/h1
汇总写到 <collect>/summary.md 并完整打印（贴这一段即可）
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
import json
import time
import argparse
import statistics

import torch
import torch.nn.functional as F

from diag_repr import extract

SCALE = 100.0
ITERS = [1, 3, 10]
MODES = ["soft", "hard", "sk_uni", "sk_sqrt"]
# (名称, 分配方式, λ, 是否中心化)
CONFIGS = [(f"{m}|λ{lam}", m, lam, False) for m in MODES for lam in (0.0, 0.5)] + \
          [(f"{m}|center", m, 0.0, True) for m in MODES]
TASKS = ["Ar2Cl", "Pr2Cl", "Re2Cl", "Cl2Ar", "Pr2Ar", "Re2Ar", "Ar2Pr", "Cl2Pr", "Re2Pr", "Ar2Re", "Cl2Re", "Pr2Re"]
COSMO = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Ar2Re": 92.0, "Cl2Ar": 85.7, "Cl2Pr": 92.4, "Cl2Re": 91.9,
         "Pr2Ar": 84.3, "Pr2Cl": 76.1, "Pr2Re": 91.7, "Re2Ar": 85.6, "Re2Cl": 75.5, "Re2Pr": 92.7}


def sinkhorn(logits, mu, iters=200):
    """行边际 1/N、列边际 mu 的熵正则 OT，核 = exp(logits)；返回每行和为 1 的分配 [N, C]"""
    N = logits.shape[0]
    log_a = torch.full((N,), -torch.log(torch.tensor(float(N))), device=logits.device, dtype=torch.float64)
    log_b = torch.log(mu.double().clamp_min(1e-12))
    L = logits.double()
    g = torch.zeros(L.shape[1], dtype=torch.float64, device=L.device)
    for _ in range(iters):
        f = log_a - torch.logsumexp(L + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(L + f.unsqueeze(1), dim=0)
    return torch.softmax(L + g.unsqueeze(0), dim=1).float()


def assign(logits, mode, mu_sqrt):
    if mode == "soft":
        return torch.softmax(logits, dim=1)
    if mode == "hard":
        return F.one_hot(logits.argmax(1), logits.shape[1]).float()
    C = logits.shape[1]
    mu = torch.full((C,), 1.0 / C, device=logits.device) if mode == "sk_uni" else mu_sqrt
    return sinkhorn(logits, mu)


def refine(X, W0, P0, mode, lam, mu_sqrt, y):
    """X: [N, d] 已归一化；W0: [C, d] 初始原型（文本先验）；P0: 第 0 轮的分配（None = 用 W0 分配）。返回 {轮数: 准确率}、最终预测"""
    W, out, P = W0, {}, P0
    for it in range(1, max(ITERS) + 1):
        if P is None:
            P = assign(SCALE * X @ W.T, mode, mu_sqrt)
        cen = F.normalize(P.T @ X, dim=1)
        W = F.normalize((1 - lam) * cen + lam * W0, dim=1) if lam > 0 else cen
        pred = (X @ W.T).argmax(1)
        if it in ITERS:
            out[it] = (pred == y).float().mean().item()
        P = None
    return out, pred


def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()
    snap, feats, q_m, q_v, y = extract(cli.snapshot, device, cli.num_workers)
    T = snap["text_feat"].to(device)
    N, C = q_v.shape
    t_pred, s_pred = q_v.argmax(1), q_m.argmax(1)
    t_ok, s_ok = t_pred == y, s_pred == y
    cnt = torch.bincount(t_pred, minlength=C).float().clamp_min(1)
    mu_sqrt = cnt.sqrt() / cnt.sqrt().sum()
    res = {"progress": snap["progress"], "iter": snap["iter"], "N": N,
           "teacher": t_ok.float().mean().item(), "student": s_ok.float().mean().item(), "clip": {}, "student_feat": {}}
    Xc = feats["clip"]
    Xc_n = F.normalize(Xc, dim=1)
    Xc_c = F.normalize(Xc - Xc.mean(0), dim=1)
    for name, mode, lam, center in CONFIGS:
        X = Xc_c if center else Xc_n
        # 第 0 轮分配：用文本原型在原始（未中心化）特征上的预测，和老师一致
        P0 = assign(SCALE * Xc_n @ T.T, mode, mu_sqrt)
        acc, pred = refine(X, T, P0, mode, lam, mu_sqrt, y)
        ok = pred == y
        res["clip"][name] = {"acc": acc, "fixed": int((ok & ~t_ok).sum()), "broke": int((~ok & t_ok).sum())}
    # 学生瓶颈特征：初始分配 = 学生概率（不用文本先验）
    Xs = feats["bottleneck"]
    cnt_s = torch.bincount(s_pred, minlength=C).float().clamp_min(1)
    mu_s = cnt_s.sqrt() / cnt_s.sqrt().sum()
    for center in (False, True):
        X = F.normalize(Xs - Xs.mean(0), dim=1) if center else F.normalize(Xs, dim=1)
        for mode in ("soft", "hard", "sk_sqrt"):
            P0 = q_m if mode == "soft" else (F.one_hot(s_pred, C).float() if mode == "hard" else
                                             sinkhorn(torch.log(q_m + 1e-8), mu_s))
            acc, pred = refine(X, None, P0, mode, 0.0, mu_s, y)
            ok = pred == y
            res["student_feat"][f"{mode}|{'center' if center else 'raw'}"] = {
                "acc": acc, "fixed": int((ok & ~s_ok).sum()), "broke": int((~ok & s_ok).sum())}
    with open(os.path.join(cli.out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)
    best = max(res["clip"].items(), key=lambda kv: kv[1]["acc"][10])
    print(f"[参考] 老师={res['teacher']:.4f}  学生={res['student']:.4f}  CLIP 侧第 10 轮最好={best[0]} {best[1]['acc'][10]:.4f}")
    print(f"[完成] 用时 {time.time() - t0:.0f} 秒")


def collect(cli):
    p2 = lambda v: "-" if v is None else f"{v * 100:.2f}"
    load = lambda root, t, st: (json.load(open(os.path.join(root, t, st, "result.json"), encoding="utf-8"))
                                if root and os.path.isfile(os.path.join(root, t, st, "result.json")) else None)
    md = ["# 类原型修正汇总（离线，不训练）", "",
          "每格 = 第 10 轮准确率相对老师（p000 = CLIP 零样本）的差（百分点）；括号内 = 纠正 / 改坏（相对老师）。",
          "探针 = H1 的 CLIP 特征线性探针（用标签，上限参考）。按目标域平均的列只对有结果的任务求。", ""]
    for stage in ("p000", "p100"):
        R = {t: load(cli.collect, t, stage) for t in TASKS}
        R = {t: r for t, r in R.items() if r}
        if not R:
            continue
        ts = list(R)
        groups = {k: [t for t in ts if t.endswith(k)] for k in ("Cl", "Ar", "Pr", "Re")}
        groups = {k: v for k, v in groups.items() if v}
        md += [f"## {stage}：CLIP 图像特征上的原型修正（老师侧）", "",
               "| 方法 | " + " | ".join(ts) + " | " + " | ".join(f"→{k} 平均" for k in groups) + " |",
               "|---|" + "---|" * (len(ts) + len(groups))]
        md.append("| 老师（绝对值） | " + " | ".join(p2(R[t]["teacher"]) for t in ts) + " | " +
                  " | ".join(f"{statistics.mean(R[t]['teacher'] for t in g) * 100:.2f}" for g in groups.values()) + " |")
        H = {t: load(cli.h1_root, t, stage) for t in ts}
        md.append("| 探针（绝对值） | " + " | ".join(p2(H[t]["layers"]["clip"]["probe"]) if H[t] else "-" for t in ts) + " | " +
                  " | ".join("-" for _ in groups) + " |")
        md.append("| COSMO（绝对值） | " + " | ".join(f"{COSMO[t]:.1f}" for t in ts) + " | " +
                  " | ".join(f"{statistics.mean(COSMO[t] for t in g):.2f}" for g in groups.values()) + " |")
        for name, *_ in CONFIGS:
            d = {t: R[t]["clip"][name]["acc"]["10"] - R[t]["teacher"] for t in ts}
            md.append(f"| {name} | " + " | ".join(f"{d[t] * 100:+.2f}（{R[t]['clip'][name]['fixed']}/{R[t]['clip'][name]['broke']}）" for t in ts)
                      + " | " + " | ".join(f"{statistics.mean(d[t] for t in g) * 100:+.2f}" for g in groups.values()) + " |")
        md += ["", f"{stage}：随轮数的变化（相对老师，第 1 / 3 / 10 轮；只列 λ = 0.5 和 center）", "",
               "| 方法 | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
        for name, _, lam, center in CONFIGS:
            if lam == 0.5 or center:
                md.append(f"| {name} | " + " | ".join(
                    "/".join(f"{(R[t]['clip'][name]['acc'][str(i)] - R[t]['teacher']) * 100:+.1f}" for i in ITERS) for t in ts) + " |")
        md += ["", f"## {stage}：学生瓶颈特征上的原型修正（相对学生，第 10 轮）", "",
               "| 方法 | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts),
               "| 学生（绝对值） | " + " | ".join(p2(R[t]["student"]) for t in ts) + " |"]
        for k in R[ts[0]]["student_feat"]:
            md.append(f"| {k} | " + " | ".join(
                f"{(R[t]['student_feat'][k]['acc']['10'] - R[t]['student']) * 100:+.2f}（{R[t]['student_feat'][k]['fixed']}/{R[t]['student_feat'][k]['broke']}）"
                for t in ts) + " |")
        md.append("")
    text = "\n".join(md) + "\n"
    with open(os.path.join(cli.collect, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print("=" * 30 + " 汇总开始（把这一段完整贴回）" + "=" * 30)
    print(text)
    print("=" * 30 + " 汇总结束 " + "=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--collect", type=str, default=None)
    parser.add_argument("--h1_root", type=str, default=None, help="H1 结果目录（读取 CLIP 探针作参考）")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli)
    else:
        run(cli)
