"""
锁死错误的结构：师生都错且预测相同的约 970 个样本，错在哪里、真实类离预测有多远（不训练，读取 run_repr.sh 的快照）

前两步诊断（archive/2026_10_18_repr）：骨干特征线性探针 85（真实标签），学生 76；但用任何无标签来源（学生 / 老师 / 一致 /
高置信样本）重拟合线性头都回到约 76，高置信样本训练的头在锁死样本上只对 7–9%，真实标签的探针却对约 52%。
说明锁死错误是系统性的：由干净样本决定的边界把它们划到错的类，只有它们自己的真实标签能纠正。
要设计纠正方法，先弄清这种系统性是什么形状（标签只用于描述）：
  1. 类对集中度   锁死错误按（真实类 → 预测类）计数：前 5 / 10 / 20 个类对占多少；覆盖 50% 需要几个类对；
                  反向（预测类 → 真实类）也出现的比例（互相混淆 vs 单向吞并）
  2. 吞并类       哪些预测类吸收了最多的锁死错误；各类的 预测数 / 真实数（> 1 = 过度预测）；学生召回率 < 50% 的类数
  3. 真实类排名   锁死样本上真实类在学生 / 老师 / 平均概率中的名次：第 2 / 前 3 / 前 5 的比例
                  （真实类常在第 2 名 → 类级的重新分配有希望；排在很后面 → 预测概率里没有信息）
  4. 子簇         pool2048 上球面 k-means（K = 520，仅用于描述）：锁死样本占多数的簇有多少、含多少锁死样本；
                  这些簇的簇内平均概率里，簇内真实多数类排第几（簇级命名能否纠正）

用法（一般由 scripts/run_locked.sh 调用）：
  python tools/diag_locked.py --snapshot runs/ReprDiag/both_snap/Ar2Cl_s42/snapshot_p100.pt --out runs/locked_results/xxx/p100
  python tools/diag_locked.py --collect runs/locked_results/xxx
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
from diag_cluster import spherical_kmeans

KMEANS_K, KMEANS_ITERS, KMEANS_SEEDS = 520, 50, 3


def true_rank(p, y):
    """真实类在概率 p 中的名次（0 = 第 1 名）"""
    return (p > p.gather(1, y.unsqueeze(1))).sum(1)


def rank_stats(r):
    n = max(len(r), 1)
    return {"top2": int((r == 1).sum()) / n, "top3": int((r <= 2).sum()) / n, "top5": int((r <= 4).sum()) / n,
            "median": float(r.float().median().item() + 1) if len(r) else float("nan")}


def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()
    snap, feats, q_m, q_v, y = extract(cli.snapshot, device, cli.num_workers)
    names = snap["args"]["class_info"]
    N, C = q_m.shape
    s_pred, t_pred = q_m.argmax(1), q_v.argmax(1)
    s_ok, t_ok = s_pred == y, t_pred == y
    locked = ~s_ok & ~t_ok & (s_pred == t_pred)
    nL = int(locked.sum())
    res = {"progress": snap["progress"], "iter": snap["iter"], "N": N, "locked": nL,
           "student": s_ok.float().mean().item(), "teacher": t_ok.float().mean().item()}

    # ---------------- 1. 类对集中度 ----------------
    pair = torch.zeros(C, C, dtype=torch.long, device=device)            # [真实类, 预测类]
    pair.index_put_((y[locked], s_pred[locked]), torch.ones(nL, dtype=torch.long, device=device), accumulate=True)
    flat = pair.flatten()
    cnt, idx = flat.sort(descending=True)
    cnt = cnt[cnt > 0]
    cum = cnt.cumsum(0).float() / max(nL, 1)
    res["pairs_n"] = int(len(cnt))
    res["pairs_share"] = {str(k): (cum[k - 1].item() if len(cum) >= k else 1.0) for k in (5, 10, 20, 50)}
    res["pairs_for_half"] = int((cum < 0.5).sum().item()) + 1 if len(cum) else 0
    # 全部样本（不只锁死）的学生混淆，用来判断反向是否也混淆
    conf_all = torch.zeros(C, C, dtype=torch.long, device=device)
    conf_all.index_put_((y, s_pred), torch.ones(N, dtype=torch.long, device=device), accumulate=True)
    nz = pair.nonzero()
    rev = conf_all[nz[:, 1], nz[:, 0]] > 0
    res["pairs_reverse_share"] = float((pair[nz[:, 0], nz[:, 1]] * rev).sum().item()) / max(nL, 1)
    res["top_pairs"] = [{"true": names[int(i) // C], "pred": names[int(i) % C], "n": int(flat[i]),
                         "reverse": int(conf_all[int(i) % C, int(i) // C])} for i in idx[:15] if flat[i] > 0]

    # ---------------- 2. 吞并类 ----------------
    pred_cnt = torch.bincount(s_pred, minlength=C)
    true_cnt = torch.bincount(y, minlength=C)
    lock_by_pred = torch.bincount(s_pred[locked], minlength=C)
    recall = (conf_all.diag().float() / true_cnt.clamp_min(1).float())
    res["recall_lt50"] = int((recall < 0.5).sum())
    res["recall_lt70"] = int((recall < 0.7).sum())
    order = lock_by_pred.argsort(descending=True)[:10]
    res["sinks"] = [{"cls": names[int(c)], "locked": int(lock_by_pred[c]), "pred": int(pred_cnt[c]), "true": int(true_cnt[c]),
                     "recall": recall[c].item()} for c in order]
    res["sink_top10_share"] = int(lock_by_pred[order].sum()) / max(nL, 1)
    worst = recall.argsort()[:10]
    res["low_recall"] = [{"cls": names[int(c)], "recall": recall[c].item(), "true": int(true_cnt[c]), "pred": int(pred_cnt[c]),
                          "locked_from": int(pair[c].sum())} for c in worst]

    # ---------------- 3. 真实类排名 ----------------
    avg = (q_m + q_v) / 2
    res["rank"] = {k: rank_stats(true_rank(p, y)[locked]) for k, p in (("student", q_m), ("teacher", q_v), ("avg", avg))}

    # ---------------- 4. 子簇 ----------------
    x = F.normalize(feats["pool2048"] - feats["pool2048"].mean(0), dim=1)
    with torch.no_grad():
        assign = max((spherical_kmeans(x, KMEANS_K, KMEANS_ITERS, s) for s in range(KMEANS_SEEDS)), key=lambda r: r[1])[0]
    oh = F.one_hot(assign, KMEANS_K).float()                               # [N, K]
    size = oh.sum(0)
    lk = oh.T @ locked.float()
    ycnt = oh.T @ F.one_hot(y, C).float()                                  # [K, C]
    purity = ycnt.max(1).values.sum().item() / N
    dom = (lk > size / 2) & (size > 0)                                     # 锁死样本占多数的簇
    mean_avg = (oh.T @ avg) / size.clamp_min(1).unsqueeze(1)
    ymaj = ycnt.argmax(1)
    crank = (mean_avg > mean_avg.gather(1, ymaj.unsqueeze(1))).sum(1)      # 簇内真实多数类在簇平均概率中的名次
    res["cluster"] = {"K": KMEANS_K, "purity": purity, "n_dom": int(dom.sum()),
                      "locked_in_dom": int(lk[dom].sum()), "dom_size_mean": size[dom].mean().item() if dom.any() else float("nan"),
                      "dom_purity": (ycnt[dom].max(1).values.sum() / size[dom].sum()).item() if dom.any() else float("nan"),
                      "rank": rank_stats(crank[dom]),
                      "locked_in_mixed": int(lk[~dom].sum())}
    with open(os.path.join(cli.out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)
    print(f"[完成] 锁死 {nL}，类对 {res['pairs_n']}，前 10 类对占 {res['pairs_share']['10']:.3f}，用时 {time.time() - t0:.0f} 秒")


def collect(cli):
    files = sorted(glob.glob(os.path.join(cli.collect, "*", "result.json")))
    if not files:
        print(f"[错误] {cli.collect} 下没有 result.json"); return
    R = sorted((json.load(open(f, encoding="utf-8")) for f in files), key=lambda r: r["iter"])
    p1 = lambda v: "-" if v != v else f"{v * 100:.1f}"
    head = "| | " + " | ".join(f"p{round(r['progress'] * 100):03d}（{r['iter']}）" for r in R) + " |"
    sep = "|---|" + "---|" * len(R)
    row = lambda lab, fn: f"| {lab} | " + " | ".join(fn(r) for r in R) + " |"
    md = ["# 锁死错误结构汇总", "", "## 总表", "", head, sep,
          row("学生 / 老师", lambda r: f"{p1(r['student'])} / {p1(r['teacher'])}"),
          row("锁死样本", lambda r: str(r["locked"])),
          row("不同的（真实 → 预测）类对数", lambda r: str(r["pairs_n"])),
          row("前 5 / 10 / 20 / 50 个类对占比", lambda r: " / ".join(p1(r["pairs_share"][k]) for k in ("5", "10", "20", "50"))),
          row("覆盖 50% 需要的类对数", lambda r: str(r["pairs_for_half"])),
          row("反向也混淆的占比", lambda r: p1(r["pairs_reverse_share"])),
          row("前 10 个吞并类占锁死的比例", lambda r: p1(r["sink_top10_share"])),
          row("学生召回率 < 50% / < 70% 的类数", lambda r: f"{r['recall_lt50']} / {r['recall_lt70']}")]
    for k, lab in (("student", "学生"), ("teacher", "老师"), ("avg", "平均")):
        md.append(row(f"真实类排名（{lab}）：第 2 / 前 3 / 前 5 / 中位名次",
                      lambda r, k=k: f"{p1(r['rank'][k]['top2'])} / {p1(r['rank'][k]['top3'])} / {p1(r['rank'][k]['top5'])} / {r['rank'][k]['median']:.0f}"))
    md += [row("子簇（K=520）纯度", lambda r: p1(r["cluster"]["purity"])),
           row("锁死占多数的簇：个数 / 含锁死 / 平均大小 / 簇纯度", lambda r: f"{r['cluster']['n_dom']} / {r['cluster']['locked_in_dom']} / "
               f"{r['cluster']['dom_size_mean']:.1f} / {p1(r['cluster']['dom_purity'])}"),
           row("这些簇：真实多数类在簇平均概率中 第 2 / 前 3 / 前 5", lambda r: " / ".join(p1(r["cluster"]["rank"][k]) for k in ("top2", "top3", "top5"))),
           row("其余锁死样本（在混合簇里）", lambda r: str(r["cluster"]["locked_in_mixed"]))]
    last = R[-1]
    md += ["", f"## 最后一个快照（p{round(last['progress'] * 100):03d}）的细节", "",
           "前 15 个锁死类对（真实 → 预测，括号内为反向在全部样本中的混淆数）：", ""]
    md += ["- " + "；".join(f"{p['true']} → {p['pred']} {p['n']}（反向 {p['reverse']}）" for p in last["top_pairs"])]
    md += ["", "吸收锁死错误最多的 10 个预测类（锁死数 / 预测数 / 真实数 / 该类召回率）：", "",
           "- " + "；".join(f"{s['cls']} {s['locked']} / {s['pred']} / {s['true']} / {p1(s['recall'])}" for s in last["sinks"]),
           "", "召回率最低的 10 个类（召回率 / 真实数 / 预测数 / 被锁死到别的类的数量）：", "",
           "- " + "；".join(f"{s['cls']} {p1(s['recall'])} / {s['true']} / {s['pred']} / {s['locked_from']}" for s in last["low_recall"])]
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
