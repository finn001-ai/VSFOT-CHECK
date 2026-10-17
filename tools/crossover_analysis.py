"""
交叉点检测：训练中不用标签，能否判断「当前阶段学生和老师谁更准」（读取 --log_samples 保存的 samples.npz，不训练）

共识塌缩实验（archive/2026_10_17_collapse）用标签发现：早期老师更准，约第 204 步后分歧样本上学生判对的次数约为老师的 2 倍。
分歧裁决分析已经说明「逐样本」分辨谁对几乎做不到；这里检验更容易的「整体」问题：
每次评估给出一个分数，> 0 表示估计学生更准，看它的符号是否和真实的 学生准确率 − 老师准确率 一致、估出的交叉点离真实交叉点多远。

参照：iter 0 的学生 = 源模型（src），iter 0 的老师 = adapter 刚初始化的 CLIP（≈ 零样本，zs）。
估计器（标签只用于评估）：
  ref_src    log A(S, src) − log A(T, src)；A 为一致率。若各方犯错相互独立，A(X, R) ≈ acc_X · acc_R，
             所以两者之比 = acc_S / acc_T，与参照自身的准确率无关。学生由源模型初始化，预计偏向学生
  ref_zs     同上，参照换成零样本 CLIP，预计偏向老师
  ref_geo    (ref_src + ref_zs) / 2，两个方向相反的偏差相互抵消
  ent_rank   分歧样本上 老师熵的百分位均值 − 学生熵的百分位均值（百分位在各自模型的全部样本内计算，消除量纲差异）
  concede    上一次评估分歧、这一次一致的样本中：(老师改成学生原预测 − 学生改成老师原预测) / 总数
  stable     老师翻转率 − 学生翻转率（相对上一次评估；更稳的一方更准）
指标：
  符号一致率   估计器符号与真实差符号相同的评估次数比例（真实差为 0 的不计）
  交叉点       第一次分数 > 0 且之后一直 > 0 的评估步数；真实交叉点同样定义（准确率差）

用法（一般由 scripts/run_crossover.sh 调用）：
  python tools/crossover_analysis.py --process runs/AblationFix --out runs/crossover_results/2026_10_17
输出：<out>/summary.md，并在「汇总开始 / 汇总结束」之间完整打印
"""

import os as _os
# 本脚本在 tools/ 下：切换到项目根目录（相对路径读 runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)

import os
import glob
import argparse
import numpy as np

ESTIMATORS = ["ref_src", "ref_zs", "ref_geo", "ent_rank", "concede", "stable"]


def rank01(x):
    """每个样本的值在本次评估所有样本中的百分位（0 = 最小）"""
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def scores(z):
    """每次评估（e ≥ 1）一行：真实准确率、一致率和各估计器分数"""
    y, s, t = z["label"], z["s_pred"], z["t_pred"]
    s_ent, t_ent = z["s_ent"], z["t_ent"]
    src, zs = s[0], t[0]
    rows = []
    for e in range(1, len(z["iter"])):
        a = lambda p, q: max((p == q).mean(), 1e-6)
        d = s[e] != t[e]
        ref_src = np.log(a(s[e], src)) - np.log(a(t[e], src))
        ref_zs = np.log(a(s[e], zs)) - np.log(a(t[e], zs))
        ent = (rank01(t_ent[e])[d] - rank01(s_ent[e])[d]).mean() if d.any() else 0.0
        resolved = (s[e - 1] != t[e - 1]) & (s[e] == t[e])
        t_moved = (resolved & (t[e] == s[e - 1])).sum()          # 老师改成学生上一次的预测
        s_moved = (resolved & (s[e] == t[e - 1])).sum()          # 学生改成老师上一次的预测
        rows.append({
            "iter": int(z["iter"][e]),
            "s_acc": (s[e] == y).mean(), "t_acc": (t[e] == y).mean(), "agree": 1 - d.mean(),
            "ref_src": ref_src, "ref_zs": ref_zs, "ref_geo": (ref_src + ref_zs) / 2, "ent_rank": ent,
            "concede": (t_moved - s_moved) / resolved.sum() if resolved.any() else 0.0,
            "stable": (t[e] != t[e - 1]).mean() - (s[e] != s[e - 1]).mean(),
        })
    return rows


def crossover(iters, vals):
    """第一次 > 0 且之后一直 > 0 的步数；从未满足返回 None"""
    for i in range(len(vals)):
        if all(v > 0 for v in vals[i:]):
            return iters[i]
    return None


def analyse(path):
    rows = scores(np.load(path))
    iters = [r["iter"] for r in rows]
    truth = [r["s_acc"] - r["t_acc"] for r in rows]
    true_x = crossover(iters, truth)
    md = [f"真实交叉点（学生准确率 − 老师准确率 之后一直 > 0）：{true_x}", "",
          "| 估计器 | 符号一致率（全部） | 符号一致率（前 10 次评估） | 估计交叉点 | 与真实相差（步） |", "|---|---|---|---|---|"]
    for k in ESTIMATORS:
        v = [r[k] for r in rows]
        ok = [np.sign(a) == np.sign(b) for a, b in zip(v, truth) if b != 0]
        ok10 = [np.sign(a) == np.sign(b) for a, b in zip(v[:10], truth[:10]) if b != 0]
        x = crossover(iters, v)
        diff = "-" if x is None or true_x is None else f"{x - true_x:+d}"
        md.append(f"| {k} | {100 * np.mean(ok):.0f}% | {100 * np.mean(ok10):.0f}% | {x} | {diff} |")
    # 前 12 次评估逐行列出（交叉点在早期），之后每 5 次一行
    pick = [r for i, r in enumerate(rows) if i < 12 or i % 5 == 4 or i == len(rows) - 1]
    md += ["", "| iter | 学生 | 老师 | 学生−老师 | 一致率 | " + " | ".join(ESTIMATORS) + " |",
           "|" + "---|" * (5 + len(ESTIMATORS))]
    for r in pick:
        md.append(f"| {r['iter']} | {100 * r['s_acc']:.2f} | {100 * r['t_acc']:.2f} | {100 * (r['s_acc'] - r['t_acc']):+.2f} | "
                  f"{100 * r['agree']:.1f} | " + " | ".join(f"{r[k]:+.4f}" for k in ESTIMATORS) + " |")
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, required=True, help="训练输出目录，如 runs/AblationFix")
    parser.add_argument("--out", type=str, required=True)
    a = parser.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.process, "*", "*", "samples.npz")))
    if not files:
        print(f"[错误] {a.process} 下没有 samples.npz（需要用 --log_dynamics --log_samples 训练）"); return

    md = ["# 交叉点检测：不用标签判断学生和老师谁更准", "",
          "分数 > 0 = 估计学生更准。估计器定义见 tools/crossover_analysis.py 开头。", ""]
    for f in files:
        name = f"{os.path.basename(os.path.dirname(os.path.dirname(f)))} / {os.path.basename(os.path.dirname(f))}"
        md += [f"## {name}", ""] + analyse(f) + [""]

    text = "\n".join(md) + "\n"
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print("=" * 30 + " 汇总开始（把这一段完整贴回）" + "=" * 30)
    print(text)
    print("=" * 30 + " 汇总结束 " + "=" * 30)


if __name__ == "__main__":
    main()
