"""
混淆类拆分诊断：表示里那 3.6 个点（Ar→Pr 学生探针 95.6 vs 学生 92.0），能否不用标签、只针对混淆类拿回来（不训练网络，读取快照）

Ar→Pr 的剩余错误多是成对语义混淆（Trash_Can→Bucket、Folder→Notebook、Marker→Pen、Computer→Laptop…），师生共享同一个错误。
以前的失败：用当前预测重拟合分类头（回到学生 ±0.6）、全局聚类 + CLIP argmax 命名（CLIP 错误成簇）。这里只在每个预测类内部拆分：
  第 1 步（可分性，标签只用于打分）：每个学生预测类内 k-means（k = 2 / 3，余弦），被误判进来的样本是否自成一团；
           上限 = 每个子簇按真实多数类命名。
  第 2 步（无标签命名）：子簇平均概率（(学生 + 老师) / 2）里预测类以外的第一名 r：
    ratio θ    子簇平均概率 m[r] / m[c] ≥ θ 时整个子簇改名为 r
    ratio θ+d  同上，且子簇特征中心离 r 类中心（各类高置信样本均值）比离 c 类中心更近
    near       只看中心距离：子簇中心离 r 更近就改名
    sample     不聚类、逐样本：样本离 r 类中心比离 c 类中心近、且 p[r] / p[c] ≥ 0.3 就改名（对照）
指标：准确率（相对学生）、纠正（学生错 → 对）、改坏（学生对 → 错）。

用法：
  python tools/diag_split.py --snapshot runs/H1Probe/cap02_snap/Ar2Pr_s42/snapshot_p100.pt --out runs/split_results/Ar2Pr_p100
汇总写到 <out>/summary.md 并完整打印（「汇总开始 / 汇总结束」之间）
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "tools"))
_os.chdir(_ROOT)

import os
import json
import time
import argparse

import torch
import torch.nn.functional as F

from diag_repr import extract

SPACES = ["bottleneck", "pool2048", "clip", "cat"]
KS = [2, 3]
THETAS = [0.1, 0.3, 0.5]
MIN_SIZE = 5           # 小于这个大小的子簇不改名


def kmeans(x, k, restarts=5, iters=50, seed=0):
    """余弦 k-means（x 已 L2 归一化），k-means++ 初始化，取簇内相似度和最大的一次；返回 [n] 簇号"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_score = None, -1e9
    n = x.shape[0]
    for _ in range(restarts):
        cent = x[torch.randint(n, (1,), generator=g)].clone()
        for _ in range(1, k):
            d = (1 - (x @ cent.T).max(1).values).clamp(min=0) ** 2
            p = (d / d.sum()).cpu() if d.sum() > 0 else torch.full((n,), 1.0 / n)
            cent = torch.cat([cent, x[torch.multinomial(p, 1, generator=g)]])
        for _ in range(iters):
            a = (x @ cent.T).argmax(1)
            new = torch.stack([F.normalize(x[a == j].mean(0), dim=0) if (a == j).any() else cent[j] for j in range(k)])
            if torch.allclose(new, cent):
                break
            cent = new
        score = (x @ cent.T).max(1).values.sum().item()
        if score > best_score:
            best, best_score = a.clone(), score
    return best


def class_centers(x, pred, conf, C):
    """各类中心：该类预测样本里置信度前 50% 的特征均值（L2 归一化）"""
    cen = torch.zeros(C, x.shape[1], device=x.device)
    for c in range(C):
        idx = (pred == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        v = conf[idx]
        keep = idx[v >= v.median()]
        cen[c] = F.normalize(x[keep].mean(0), dim=0)
    return cen


def score(new_pred, s_pred, y):
    ok, s_ok = new_pred == y, s_pred == y
    return {"acc": ok.float().mean().item(), "d": (ok.float().mean() - s_ok.float().mean()).item(),
            "fixed": int((ok & ~s_ok).sum()), "broke": int((~ok & s_ok).sum()), "changed": int((new_pred != s_pred).sum())}


def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()
    snap, feats, q_m, q_v, y = extract(cli.snapshot, device, cli.num_workers)
    N, C = q_m.shape
    names = snap["args"]["class_info"]
    s_pred = q_m.argmax(1)
    avg = (q_m + q_v) / 2
    conf = avg.max(1).values
    s_ok = s_pred == y
    res = {"snapshot": cli.snapshot, "iter": snap["iter"], "N": N, "student": s_ok.float().mean().item(),
           "teacher": (q_v.argmax(1) == y).float().mean().item(), "spaces": {}}
    print(f"[参考] 学生={res['student']:.4f}  老师={res['teacher']:.4f}  错误数={int((~s_ok).sum())}")

    # 混淆对（真实 a → 预测 b），只用于打分
    pairs = {}
    for a, b in zip(y[~s_ok].tolist(), s_pred[~s_ok].tolist()):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:10]

    xs = {k: F.normalize(feats[k], dim=1) for k in ("bottleneck", "pool2048", "clip")}
    xs["cat"] = F.normalize(torch.cat([xs["pool2048"], xs["clip"]], dim=1), dim=1)
    for sp in SPACES:
        x = xs[sp]
        cen = class_centers(x, s_pred, conf, C)
        out = {}
        for k in KS:
            sub = torch.full((N,), -1, dtype=torch.long, device=device)      # 全局子簇号 = 类 * k + 簇
            for c in range(C):
                idx = (s_pred == c).nonzero(as_tuple=True)[0]
                if len(idx) < 2 * MIN_SIZE:
                    sub[idx] = c * k
                    continue
                sub[idx] = c * k + kmeans(x[idx], k).to(device)
            # 第 1 步：上限（子簇按真实多数类命名）与混淆对的成团程度
            oracle = s_pred.clone()
            info = {}
            for g in sub.unique().tolist():
                m = sub == g
                maj = y[m].mode().values.item()
                if m.sum() >= MIN_SIZE:
                    oracle[m] = maj
                c = g // k
                mp = avg[m].mean(0)
                mp_c = mp.clone(); mp_c[c] = -1
                r = mp_c.argmax().item()
                d_r = (F.normalize(x[m].mean(0), dim=0) @ cen[r]).item()
                d_c = (F.normalize(x[m].mean(0), dim=0) @ cen[c]).item()
                info[g] = (int(m.sum().item()), c, r, (mp[r] / mp[c]).item(), d_r > d_c)
            out[f"k{k}_oracle"] = score(oracle, s_pred, y)
            pair_rows = []
            for (a, b), n_ab in top_pairs:
                m_ab = (y == a) & (s_pred == b)
                # a 样本落在「a 占多数」的子簇里的比例
                in_a = sum(int(((sub == g) & m_ab).sum()) for g in sub[m_ab].unique().tolist()
                           if y[sub == g].mode().values.item() == a)
                pair_rows.append((names[a], names[b], n_ab, in_a))
            out[f"k{k}_pairs"] = pair_rows
            # 第 2 步：无标签命名规则
            for th in THETAS:
                for use_d in (False, True):
                    new = s_pred.clone()
                    for g, (n, c, r, ratio, closer) in info.items():
                        if n >= MIN_SIZE and ratio >= th and (closer or not use_d):
                            new[sub == g] = r
                    out[f"k{k}_ratio{th}" + ("+d" if use_d else "")] = score(new, s_pred, y)
            new = s_pred.clone()
            for g, (n, c, r, ratio, closer) in info.items():
                if n >= MIN_SIZE and closer:
                    new[sub == g] = r
            out[f"k{k}_near"] = score(new, s_pred, y)
        # 逐样本对照
        a2 = avg.clone(); a2[torch.arange(N), s_pred] = -1
        r_s = a2.argmax(1)
        closer = (x * cen[r_s]).sum(1) > (x * cen[s_pred]).sum(1)
        ratio = avg[torch.arange(N), r_s] / avg[torch.arange(N), s_pred]
        out["sample"] = score(torch.where(closer & (ratio >= 0.3), r_s, s_pred), s_pred, y)
        res["spaces"][sp] = out
        print(f"[{sp}] 上限 k2 {out['k2_oracle']['d'] * 100:+.2f} / k3 {out['k3_oracle']['d'] * 100:+.2f}；"
              + "  ".join(f"{n} {v['d'] * 100:+.2f}({v['fixed']}/{v['broke']})" for n, v in out.items()
                          if isinstance(v, dict) and "oracle" not in n))
    with open(os.path.join(cli.out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)

    # 汇总
    p2 = lambda v: f"{v * 100:.2f}"
    md = ["# 混淆类拆分诊断", "", f"快照 `{cli.snapshot}`（iter {snap['iter']}）：学生 {p2(res['student'])}，老师 {p2(res['teacher'])}，"
          f"错误 {int((~s_ok).sum())} / {N}。每格：准确率差（百分点）/ 纠正 / 改坏 / 改动样本数。", ""]
    rules = ["oracle"] + [f"ratio{t}" + s for t in THETAS for s in ("", "+d")] + ["near"]
    for k in KS:
        md += [f"## k = {k}", "", "| 规则 | " + " | ".join(SPACES) + " |", "|---|" + "---|" * len(SPACES)]
        for rl in rules:
            md.append(f"| {rl}{'（真实标签上限）' if rl == 'oracle' else ''} | " + " | ".join(
                (lambda v: f"{v['d'] * 100:+.2f} / {v['fixed']} / {v['broke']} / {v['changed']}")(res['spaces'][sp][f'k{k}_{rl}'])
                for sp in SPACES) + " |")
        md += ["", f"混淆对成团程度（k = {k}）：真实 a 被判成 b 的样本里，落在「a 占多数」子簇的个数", "",
               "| 真实 → 预测 | 个数 | " + " | ".join(SPACES) + " |", "|---|---|" + "---|" * len(SPACES)]
        for i, (a, b, n_ab, _) in enumerate(res["spaces"][SPACES[0]][f"k{k}_pairs"]):
            md.append(f"| {a} → {b} | {n_ab} | " + " | ".join(str(res['spaces'][sp][f'k{k}_pairs'][i][3]) for sp in SPACES) + " |")
        md.append("")
    md += ["## 逐样本对照（不聚类）", "", "| " + " | ".join(SPACES) + " |", "|" + "---|" * len(SPACES),
           "| " + " | ".join((lambda v: f"{v['d'] * 100:+.2f} / {v['fixed']} / {v['broke']} / {v['changed']}")(res['spaces'][sp]['sample'])
                             for sp in SPACES) + " |", "", f"用时 {time.time() - t0:.0f} 秒"]
    text = "\n".join(md) + "\n"
    with open(os.path.join(cli.out, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print("=" * 30 + " 汇总开始 " + "=" * 30)
    print(text)
    print("=" * 30 + " 汇总结束 " + "=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    run(parser.parse_args())
