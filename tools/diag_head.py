"""
无标签重拟合分类头：表示里那 9 个点（线性探针 85 vs 学生 76），不用标签能拿回多少（不训练网络，读取 run_repr.sh 的快照）

表示诊断（archive/2026_10_18_repr）：骨干 2048 维特征的线性探针 85.0、瓶颈 84.3，学生只有 76.1，且到训练结束都不下降；
骨干有效秩不降，瓶颈有效秩 210 → 157。信息在表示里，分类器没用上。探针用了真实标签，这里把标签换成无标签来源，
在冻结特征上重新拟合一个线性头（多类逻辑回归，LBFGS，特征 z-score），看能否高于学生：
  student      学生 argmax，全部样本（对照：应≈学生）
  teacher      老师 argmax，全部样本
  soft_avg     (学生概率 + 老师概率) / 2 作软标签，全部样本
  agree        只用师生一致的样本，标签 = 共同预测
  agree_conf   一致且平均置信度不低于该类（一致样本内）中位数的样本
  conf50       每个预测类（平均概率 argmax）取平均置信度前 50% 的样本
  agree_r2     先按 agree 拟合，再用它对全部样本的预测作标签重拟合一轮
直推设定（SFDA 本来就在目标域全部样本上评估），训练用全部特征，标签只用于评估。
权重衰减报告 3 个值；方法里只能用无标签方式选，所以要看三个值是否都稳定。
指标：准确率（相对学生）、锁死样本上的准确率、救回的「两边都错」/ 改坏的「学生对」（wd = 1e-2）。

用法（一般由 scripts/run_head.sh 调用）：
  python tools/diag_head.py --snapshot runs/ReprDiag/both_snap/Ar2Cl_s42/snapshot_p050.pt --out runs/head_results/xxx/p050
  python tools/diag_head.py --collect runs/head_results/xxx
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
import torch.nn.functional as F

from diag_repr import extract

LAYERS = ["pool2048", "bottleneck"]
TARGETS = ["student", "teacher", "soft_avg", "agree", "agree_conf", "conf50", "agree_r2"]
WDS = [1e-3, 1e-2, 1e-1]
MAIN_WD = 1e-2


def fit_head(x, target, mask, wd, C):
    """x: [N, d] 已 z-score；target: [N] 硬标签或 [N, C] 软标签；只在 mask 内训练；返回全部样本的预测"""
    W = torch.zeros(x.shape[1], C, device=x.device, requires_grad=True)
    b = torch.zeros(C, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
    xt, tt = x[mask], target[mask]

    def closure():
        opt.zero_grad()
        logits = xt @ W + b
        if tt.dim() == 1:
            loss = F.cross_entropy(logits, tt)
        else:
            loss = -(tt * F.log_softmax(logits, dim=1)).sum(1).mean()
        loss = loss + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (x @ W + b).argmax(1)


def per_class_top(conf, cls, C, frac=None, median=False):
    """每个类内按置信度取前 frac（或不低于中位数）的样本；返回 bool mask"""
    keep = torch.zeros_like(cls, dtype=torch.bool)
    for c in range(C):
        idx = (cls == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        v = conf[idx]
        thr = v.median() if median else v.sort(descending=True).values[max(int(len(idx) * frac) - 1, 0)]
        keep[idx[v >= thr]] = True
    return keep


def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()
    snap, feats, q_m, q_v, y = extract(cli.snapshot, device, cli.num_workers)
    N, C = q_m.shape
    s_pred, t_pred = q_m.argmax(1), q_v.argmax(1)
    s_ok, t_ok = s_pred == y, t_pred == y
    locked = ~s_ok & ~t_ok & (s_pred == t_pred)
    avg = (q_m + q_v) / 2
    agree = s_pred == t_pred
    all_ = torch.ones(N, dtype=torch.bool, device=device)
    # agree_conf：一致样本内，每个类按平均置信度取不低于中位数的
    conf = avg.max(1).values
    ac = torch.zeros(N, dtype=torch.bool, device=device)
    ai = agree.nonzero(as_tuple=True)[0]
    ac[ai[per_class_top(conf[ai], s_pred[ai], C, median=True)]] = True
    c50 = per_class_top(conf, avg.argmax(1), C, frac=0.5)
    sets = {"student": (s_pred, all_), "teacher": (t_pred, all_), "soft_avg": (avg, all_),
            "agree": (s_pred, agree), "agree_conf": (s_pred, ac), "conf50": (avg.argmax(1), c50)}

    res = {"progress": snap["progress"], "iter": snap["iter"], "N": N,
           "student": s_ok.float().mean().item(), "teacher": t_ok.float().mean().item(),
           "union": (s_ok | t_ok).float().mean().item(), "locked": int(locked.sum()),
           "n_train": {**{k: int(m.sum()) for k, (_, m) in sets.items()}, "agree_r2": N},
           "label_acc": {k: ((t if t.dim() == 1 else t.argmax(1))[m] == y[m]).float().mean().item() for k, (t, m) in sets.items()},
           "layers": {}}
    print(f"[参考] 学生={res['student']:.4f}  老师={res['teacher']:.4f}  并集={res['union']:.4f}  锁死={res['locked']}")
    for L in LAYERS:
        x = feats[L]
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        out = {}
        with torch.enable_grad():
            for wd in WDS:
                preds = {k: fit_head(x, t, m, wd, C) for k, (t, m) in sets.items()}
                preds["agree_r2"] = fit_head(x, preds["agree"], all_, wd, C)
                for k, p in preds.items():
                    ok = p == y
                    out.setdefault(k, {})[str(wd)] = {
                        "acc": ok.float().mean().item(), "d_student": ok.float().mean().item() - res["student"],
                        "locked_acc": (ok[locked].float().mean().item() if locked.any() else float("nan")),
                        "saved": int((ok & ~s_ok & ~t_ok).sum()), "broke": int((~ok & s_ok).sum())}
        res["layers"][L] = out
        print(f"[{L}] " + "  ".join(f"{k}={out[k][str(MAIN_WD)]['acc']:.4f}" for k in TARGETS))
    with open(os.path.join(cli.out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)
    print(f"[完成] 用时 {time.time() - t0:.0f} 秒")


def collect(cli):
    files = sorted(glob.glob(os.path.join(cli.collect, "*", "result.json")))
    if not files:
        print(f"[错误] {cli.collect} 下没有 result.json"); return
    R = sorted((json.load(open(f, encoding="utf-8")) for f in files), key=lambda r: r["iter"])
    p2 = lambda v: "-" if v != v else f"{v * 100:.2f}"
    md = ["# 无标签重拟合分类头汇总", "",
          "线性探针（真实标签，5 折 CV）参考：pool2048 85.02 / bottleneck 84.26（p100）。", ""]
    for r in R:
        md += [f"## p{round(r['progress'] * 100):03d}（iter {r['iter']}）：学生 {p2(r['student'])}，老师 {p2(r['teacher'])}，"
               f"并集 {p2(r['union'])}，锁死 {r['locked']}", "",
               "训练标签：样本数 / 标签准确率 —— " + "，".join(
                   f"{k} {r['n_train'][k]} / {p2(r['label_acc'][k])}" for k in r["label_acc"]), "",
               "| 层 | 目标 | wd=1e-3 | wd=1e-2 | wd=1e-1 | 锁死上准确率 | 救回两边都错 / 改坏学生对 |",
               "|---|---|---|---|---|---|---|"]
        for L in LAYERS:
            for k in TARGETS:
                c = r["layers"][L][k]
                m = c[str(MAIN_WD)]
                md.append(f"| {L} | {k} | " + " | ".join(f"{p2(c[str(w)]['acc'])}（{c[str(w)]['d_student'] * 100:+.2f}）" for w in WDS)
                          + f" | {p2(m['locked_acc'])} | {m['saved']} / {m['broke']} |")
        md.append("")
    md.append("括号内为相对学生的差（百分点）；最后两列为 wd = 1e-2。")
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
