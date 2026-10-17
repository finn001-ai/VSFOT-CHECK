"""
分歧裁决分析：学生和老师意见不一致时，能否不用标签分辨谁对（读取 --log_samples 保存的 samples.npz，不训练）

共识塌缩实验（archive/2026_10_17_collapse）发现：并集峰值后丢失的样本，多数是「只有学生对」被对齐损失拉向老师的错误标签；
但同一个方向也让学生学会了「只有老师对」的样本。要只保留好的那一半，就要在分歧样本上分辨谁对。
本脚本在每次评估的分歧样本（学生预测 ≠ 老师预测）上检验几种无标签信号，标签只用于评估：
  always_student / always_teacher   总听学生 / 总听老师（对照）
  ent_raw      熵小的一方（两边熵的量纲不同，仅作参考）
  ent_rank     熵在各自模型内的百分位更低的一方
  stab         当前预测已连续保持更多次评估的一方（相同则听学生）
  src          学生当前预测 = 源模型（iter 0 的学生）的预测 → 听学生，否则听老师
  zs           老师当前预测 = 零样本 CLIP（iter 0 的老师）的预测 → 听老师，否则听学生
  vote3        ent_rank、stab、src 三者多数票
  oracle       任一方对即算对（上限）
指标：
  acc          该规则在分歧样本上选中正确一方的比例（两边都错的分歧样本计为错）
  保学生       「只有学生对」的样本中，规则选了学生的比例（越高，被拉错的越少）
  从老师学     「只有老师对」的样本中，规则选了老师的比例（越高，能从老师学到的越多）

用法（一般由 scripts/run_arbitration.sh 调用）：
  python tools/arbitration_analysis.py --process runs/AblationFix --out runs/arbitration_results/2026_10_17

输出：<out>/arbitration.md（同时打印）和 <out>.zip
"""

import os as _os, sys as _sys
# 本脚本在 tools/ 下：切换到项目根目录（相对路径读 runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)

import os
import glob
import shutil
import argparse
import numpy as np

RULES = ["always_student", "always_teacher", "ent_raw", "ent_rank", "stab", "src", "zs", "vote3", "oracle"]


def run_length(pred):
    """每次评估时，当前预测已连续保持的评估次数（第一次为 0）"""
    out = np.zeros_like(pred)
    for e in range(1, len(pred)):
        out[e] = np.where(pred[e] == pred[e - 1], out[e - 1] + 1, 0)
    return out


def rank01(x):
    """每个样本的值在本次评估所有样本中的百分位（0 = 最小）"""
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def choose_student(rule, e, d, s, t, s_ent, t_ent, s_run, t_run):
    """规则在第 e 次评估、分歧样本 d 上是否选学生（True = 听学生）"""
    if rule == "always_student":
        return np.ones(d.sum(), bool)
    if rule == "always_teacher":
        return np.zeros(d.sum(), bool)
    if rule == "ent_raw":
        return s_ent[e][d] <= t_ent[e][d]
    if rule == "ent_rank":
        return rank01(s_ent[e])[d] <= rank01(t_ent[e])[d]
    if rule == "stab":
        return s_run[e][d] >= t_run[e][d]
    if rule == "src":
        return s[e][d] == s[0][d]
    if rule == "zs":
        return t[e][d] != t[0][d]
    if rule == "vote3":
        v = sum(choose_student(r, e, d, s, t, s_ent, t_ent, s_run, t_run).astype(int) for r in ("ent_rank", "stab", "src"))
        return v >= 2
    raise ValueError(rule)


def analyse(path):
    z = np.load(path)
    iters, y = z["iter"], z["label"]
    s, t, s_ent, t_ent = z["s_pred"], z["t_pred"], z["s_ent"], z["t_ent"]
    s_run, t_run = run_length(s), run_length(t)
    rows = []                                      # 每次评估一行：{rule: (acc, 保学生, 从老师学)}，以及分歧样本构成
    for e in range(1, len(iters)):
        d = s[e] != t[e]
        if not d.any():
            continue
        s_ok, t_ok = (s[e] == y)[d], (t[e] == y)[d]
        row = {"iter": int(iters[e]), "n": int(d.sum()), "s_only": int((s_ok & ~t_ok).sum()), "t_only": int((t_ok & ~s_ok).sum())}
        for r in RULES:
            if r == "oracle":
                row[r] = ((s_ok | t_ok).mean(), 1.0, 1.0)
                continue
            cs = choose_student(r, e, d, s, t, s_ent, t_ent, s_run, t_run)
            acc = np.where(cs, s_ok, t_ok).mean()
            keep = cs[s_ok & ~t_ok].mean() if (s_ok & ~t_ok).any() else float("nan")
            learn = (~cs)[t_ok & ~s_ok].mean() if (t_ok & ~s_ok).any() else float("nan")
            row[r] = (acc, keep, learn)
        rows.append(row)
    return rows


def fmt(v):
    return "-" if v != v else f"{v * 100:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, required=True, help="训练输出目录，如 runs/AblationFix")
    parser.add_argument("--out", type=str, required=True)
    a = parser.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.process, "*", "*", "samples.npz")))
    if not files:
        print(f"[错误] {a.process} 下没有 samples.npz（需要用 --log_dynamics --log_samples 训练）"); return

    md = ["# 分歧裁决分析", "",
          "只看学生预测 ≠ 老师预测的样本。每格为「acc / 保学生 / 从老师学」（%）：",
          "acc = 选中正确一方的比例；保学生 = 只有学生对的样本中选了学生的比例；从老师学 = 只有老师对的样本中选了老师的比例。",
          "规则说明见 tools/arbitration_analysis.py 开头。", ""]
    for f in files:
        name = f"{os.path.basename(os.path.dirname(os.path.dirname(f)))} / {os.path.basename(os.path.dirname(f))}"
        rows = analyse(f)
        if not rows:
            continue
        # 后期汇总：第 612 步（并集峰值附近）之后的所有评估，按分歧样本数加权
        late = [r for r in rows if r["iter"] >= 612]
        md += [f"## {name}", "", "### 后期汇总（iter ≥ 612，按分歧样本数加权）", "",
               "| 规则 | acc | 保学生 | 从老师学 |", "|---|---|---|---|"]
        for rule in RULES:
            w = np.array([r["n"] for r in late], float)
            ws = np.array([r["s_only"] for r in late], float)
            wt = np.array([r["t_only"] for r in late], float)
            acc = np.average([r[rule][0] for r in late], weights=w)
            keep = np.nansum([r[rule][1] * k for r, k in zip(late, ws)]) / max(ws.sum(), 1)
            learn = np.nansum([r[rule][2] * k for r, k in zip(late, wt)]) / max(wt.sum(), 1)
            md.append(f"| {rule} | {fmt(acc)} | {fmt(keep)} | {fmt(learn)} |")
        pick = [r for r in rows if r["iter"] in (136, 340, 612, 1020, 1360, 1700, 2380)] + [rows[-1]]
        md += ["", "### 各阶段", "",
               "| iter | 分歧样本 | 只有学生对 | 只有老师对 | " + " | ".join(RULES) + " |",
               "|" + "---|" * (4 + len(RULES))]
        for r in pick:
            md.append(f"| {r['iter']} | {r['n']} | {r['s_only']} | {r['t_only']} | "
                      + " | ".join("/".join(fmt(v) for v in r[rule]) for rule in RULES) + " |")
        md.append("")

    text = "\n".join(md) + "\n"
    with open(os.path.join(a.out, "arbitration.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print(text)
    zip_path = shutil.make_archive(a.out.rstrip("/"), "zip", root_dir=a.out)
    print(f"已打包: {zip_path}")


if __name__ == "__main__":
    main()
