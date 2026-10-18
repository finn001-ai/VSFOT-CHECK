"""
类边际纠正：锁死错误是不是「过度预测的类吞并了真实类排第 2 的样本」（不训练，读取 run_repr.sh 的快照）

锁死错误结构（archive/2026_10_18_repr）：错误分散在 381 个类对上（不能逐对修），但多数是单向吞并（反向也混淆的只占 36%），
吸收错误的类明显过度预测（Calculator 预测 90 / 真实 46，Lamp_Shade 74 / 40，Flipflops 69 / 40），
而锁死样本的真实类有 47% 排在第 2、73% 在前 5。若如此，问题不在 970 个样本，而在 65 个类的边际。
检验：把全部样本的概率 P 在给定类边际 μ 下重新分配（全数据集 Sinkhorn，代价 −log P；ε = 1 时等价于按 μ 做先验重加权），
比较 μ 的几种来源（oracle 只作上界）：
  pred      学生 argmax 计数（对照：≈ 不改变）
  mean_p    平均概率
  temp05    预测计数的平方根（向均匀收缩一半）
  uniform   均匀
  true      真实类频率（用了标签，上界）
概率 P 用 学生 / (学生 + 老师) / 2 两种。报告准确率、相对学生、锁死样本被纠正数、改坏的「学生对」，以及 μ 与真实边际的 L1 距离。

用法（一般由 scripts/run_marginal.sh 调用）：
  python tools/diag_marginal.py --snapshot runs/ReprDiag/both_snap/Ar2Cl_s42/snapshot_p100.pt --out runs/marginal_results/xxx/p100
  python tools/diag_marginal.py --collect runs/marginal_results/xxx
汇总写到 <collect>/summary.md 并完整打印（贴这一段即可）
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
import glob
import json
import time
import argparse

import torch

from diag_repr import extract

MARGINALS = ["pred", "mean_p", "temp05", "uniform", "true"]
EPS = [1.0, 0.5]


def sinkhorn_assign(P, mu, eps, iters=1000):
    """全数据集 Sinkhorn：行边际 1/N，列边际 mu，代价 −log P；返回每个样本计划最大的类"""
    N = P.shape[0]
    logK = torch.log(P.double() + 1e-12) / eps                     # −cost / ε
    log_a = torch.full((N,), -torch.log(torch.tensor(float(N))), dtype=torch.float64, device=P.device)
    log_b = torch.log(mu.double().clamp_min(1e-12))
    g = torch.zeros(P.shape[1], dtype=torch.float64, device=P.device)
    for _ in range(iters):
        f = log_a - torch.logsumexp(logK + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(logK + f.unsqueeze(1), dim=0)
    return (logK + f.unsqueeze(1) + g.unsqueeze(0)).argmax(1)


def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()
    snap, _, q_m, q_v, y = extract(cli.snapshot, device, cli.num_workers)
    N, C = q_m.shape
    s_pred, t_pred = q_m.argmax(1), q_v.argmax(1)
    s_ok, t_ok = s_pred == y, t_pred == y
    locked = ~s_ok & ~t_ok & (s_pred == t_pred)
    true_mu = torch.bincount(y, minlength=C).double() / N
    res = {"progress": snap["progress"], "iter": snap["iter"], "N": N, "locked": int(locked.sum()),
           "student": s_ok.float().mean().item(), "teacher": t_ok.float().mean().item(),
           "true_min_max": [int(true_mu.min() * N), int(true_mu.max() * N)], "rows": {}, "l1": {}}
    for pname, P in (("student", q_m), ("avg", (q_m + q_v) / 2)):
        cnt = torch.bincount(P.argmax(1), minlength=C).double()
        mus = {"pred": cnt / N, "mean_p": P.mean(0).double(), "temp05": cnt.sqrt() / cnt.sqrt().sum(),
               "uniform": torch.full((C,), 1.0 / C, dtype=torch.float64, device=device), "true": true_mu}
        res["l1"][pname] = {k: (m - true_mu).abs().sum().item() for k, m in mus.items()}
        base_ok = P.argmax(1) == y
        res["rows"][f"{pname}_argmax"] = {"acc": base_ok.float().mean().item(), "d": base_ok.float().mean().item() - res["student"],
                                          "fixed": int((base_ok & locked).sum()), "broke": int((~base_ok & s_ok).sum())}
        for mk, mu in mus.items():
            for eps in EPS:
                ok = sinkhorn_assign(P, mu, eps) == y
                res["rows"][f"{pname}|{mk}|{eps}"] = {"acc": ok.float().mean().item(), "d": ok.float().mean().item() - res["student"],
                                                      "fixed": int((ok & locked).sum()), "broke": int((~ok & s_ok).sum())}
    with open(os.path.join(cli.out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)
    print(f"[完成] 用时 {time.time() - t0:.0f} 秒")


def collect(cli):
    files = sorted(glob.glob(os.path.join(cli.collect, "*", "result.json")))
    if not files:
        print(f"[错误] {cli.collect} 下没有 result.json"); return
    R = sorted((json.load(open(f, encoding="utf-8")) for f in files), key=lambda r: r["iter"])
    p2 = lambda v: f"{v * 100:.2f}"
    cell = lambda c: f"{p2(c['acc'])}（{c['d'] * 100:+.2f}） {c['fixed']}/{c['broke']}"
    head = "| 概率 | 边际 μ | ε | " + " | ".join(f"p{round(r['progress'] * 100):03d}（{r['iter']}）" for r in R) + " |"
    md = ["# 类边际纠正汇总", "",
          "每格：准确率（相对学生） 纠正的锁死样本 / 改坏的学生对。true = 真实类频率（用了标签，上界）。", "",
          "| | " + " | ".join(f"p{round(r['progress'] * 100):03d}" for r in R) + " |", "|---|" + "---|" * len(R),
          "| 学生 / 老师 / 锁死 | " + " | ".join(f"{p2(r['student'])} / {p2(r['teacher'])} / {r['locked']}" for r in R) + " |",
          "| 真实类样本数 最少 / 最多 | " + " | ".join(f"{r['true_min_max'][0]} / {r['true_min_max'][1]}" for r in R) + " |", "",
          head, "|---|---|---|" + "---|" * len(R)]
    for pname in ("student", "avg"):
        md.append(f"| {pname} | argmax | - | " + " | ".join(cell(r["rows"][f"{pname}_argmax"]) for r in R) + " |")
        for mk in MARGINALS:
            for eps in EPS:
                md.append(f"| {pname} | {mk} | {eps} | " + " | ".join(cell(r["rows"][f"{pname}|{mk}|{eps}"]) for r in R) + " |")
    md += ["", "边际估计与真实边际的 L1 距离（0 = 完全一致，最大 2）：", "",
           "| 概率 | " + " | ".join(MARGINALS) + " |", "|---|" + "---|" * len(MARGINALS)]
    for pname in ("student", "avg"):
        md.append(f"| {pname}（p{round(R[-1]['progress'] * 100):03d}） | " + " | ".join(f"{R[-1]['l1'][pname][k]:.3f}" for k in MARGINALS) + " |")
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli)
    else:
        run(cli)
