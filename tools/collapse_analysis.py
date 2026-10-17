"""
共识塌缩分析：读取 --log_samples 保存的逐样本预测（samples.npz），统计学生 / 老师的状态转移

每个样本在每次评估时处于 4 种状态之一：
  都对（学生、老师都判对）/ 只有学生对 / 只有老师对 / 都错
回答的问题：
  1. 并集（任一判对）在哪次评估最高，之后丢了多少样本、救回多少样本；
  2. 丢掉的样本在并集最高点处于什么状态：
     「只有学生对」→ 后来都错，且学生最终预测 = 老师在峰值时的预测：学生吸收了老师的错误
     「只有老师对」→ 后来都错，且老师最终预测 = 学生在峰值时的预测：老师吸收了学生的错误
  3. 最终都错的样本里，有多少在训练中曾被某一方判对过（原则上可救回）

用法（一般由 scripts/run_collapse.sh 调用）：
  python tools/collapse_analysis.py --process runs/AblationFix --out runs/ablation_results/2026_10_17

输出：<out>/collapse.md（同时打印）
"""

import os as _os, sys as _sys
# 本脚本在 tools/ 下：切换到项目根目录（相对路径读 runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)

import os
import glob
import argparse
import numpy as np

STATES = ["都错", "只有老师对", "只有学生对", "都对"]      # 编码 = 2 * 学生对 + 老师对


def analyse(path):
    d = np.load(path)
    iters, y = d["iter"], d["label"]
    s, t = d["s_pred"], d["t_pred"]                       # [E, N]
    s_ok, t_ok = s == y, t == y
    state = 2 * s_ok.astype(int) + t_ok.astype(int)       # [E, N]
    union = (state > 0).sum(1)
    pk, fn = int(union.argmax()), len(iters) - 1
    lost = (state[pk] > 0) & (state[fn] == 0)
    gained = (state[pk] == 0) & (state[fn] > 0)
    lost_s = lost & (state[pk] == 2)                      # 峰值时只有学生对
    lost_t = lost & (state[pk] == 1)                      # 峰值时只有老师对
    final_ff = state[fn] == 0
    return {
        "N": len(y), "iters": iters, "state": state, "pk": pk, "fn": fn, "union": union,
        "s_acc": s_ok.mean(1), "t_acc": t_ok.mean(1), "agree": (s == t).mean(1),
        "lost": int(lost.sum()), "gained": int(gained.sum()),
        "lost_s": int(lost_s.sum()), "lost_t": int(lost_t.sum()), "lost_tt": int((lost & (state[pk] == 3)).sum()),
        # 学生吸收老师错误：学生最终预测 = 老师在峰值时的（错误）预测
        "lost_s_absorb": int((lost_s & (s[fn] == t[pk])).sum()),
        # 老师吸收学生错误：老师最终预测 = 学生在峰值时的（错误）预测
        "lost_t_absorb": int((lost_t & (t[fn] == s[pk])).sum()),
        "final_ff": int(final_ff.sum()),
        "final_ff_agree": int((final_ff & (s[fn] == t[fn])).sum()),
        "final_ff_ever": int((final_ff & (state > 0).any(0)).sum()),
        "final_ff_always": int((state == 0).all(0).sum()),
        "trans": np.array([[int(((state[pk] == a) & (state[fn] == b)).sum()) for b in range(4)] for a in range(4)]),
    }


def pct(x):
    return f"{x * 100:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, required=True, help="训练输出目录，如 runs/AblationFix")
    parser.add_argument("--out", type=str, required=True)
    a = parser.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.process, "*", "*", "samples.npz")))
    if not files:
        print(f"[提示] {a.process} 下没有 samples.npz，跳过共识塌缩分析"); return
    res = {f"{os.path.basename(os.path.dirname(os.path.dirname(f)))} / {os.path.basename(os.path.dirname(f))}": analyse(f)
           for f in files}

    md = ["# 共识塌缩分析", "",
          "状态：都对 / 只有学生对 / 只有老师对 / 都错（老师 = 带 adapter 的 CLIP）。"
          "峰值 = 并集（任一判对）最高的那次评估；丢失 = 峰值时并集内、最终都错；救回 = 峰值时都错、最终并集内。", "",
          "## 总表", "",
          "| 组 | 最终学生 | 最终老师 | 最终一致率 | 并集峰值（iter） | 最终并集 | 丢失 | 救回 | "
          "丢失中峰值只有学生对（其中学生吸收老师错误） | 丢失中峰值只有老师对（其中老师吸收学生错误） | 丢失中峰值都对 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        pk, fn, N = r["pk"], r["fn"], r["N"]
        md.append(f"| {name} | {pct(r['s_acc'][fn])} | {pct(r['t_acc'][fn])} | {pct(r['agree'][fn])} | "
                  f"{pct(r['union'][pk] / N)}（{r['iters'][pk]}） | {pct(r['union'][fn] / N)} | {r['lost']} | {r['gained']} | "
                  f"{r['lost_s']}（{r['lost_s_absorb']}） | {r['lost_t']}（{r['lost_t_absorb']}） | {r['lost_tt']} |")
    md += ["", "## 最终都错的样本", "",
           "| 组 | 最终都错 | 其中两边预测相同（锁死） | 训练中曾被某一方判对过 | 从头到尾都错 |", "|---|---|---|---|---|"]
    for name, r in res.items():
        md.append(f"| {name} | {r['final_ff']} | {r['final_ff_agree']} | {r['final_ff_ever']} | {r['final_ff_always']} |")

    for name, r in res.items():
        md += ["", f"## {name}", "", f"峰值（iter {r['iters'][r['pk']]}）→ 最终（iter {r['iters'][r['fn']]}）的状态转移（样本数，行 = 峰值状态，列 = 最终状态）：", "",
               "| 峰值 \\ 最终 | " + " | ".join(STATES) + " |", "|---|---|---|---|---|"]
        for i in range(4):
            md.append(f"| {STATES[i]} | " + " | ".join(str(v) for v in r["trans"][i]) + " |")
        md += ["", "各状态样本数随训练变化：", "",
               "| iter | 学生 | 老师 | 一致率 | " + " | ".join(STATES) + " |", "|---|---|---|---|---|---|---|---|"]
        E = len(r["iters"])
        for e in sorted({0, 1, 2, 3, 5, 8, 11, 15, 20, 25, 30, 35, E - 1, r["pk"]}):
            if e < E:
                cnt = [int((r["state"][e] == k).sum()) for k in range(4)]
                md.append(f"| {r['iters'][e]} | {pct(r['s_acc'][e])} | {pct(r['t_acc'][e])} | {pct(r['agree'][e])} | "
                          + " | ".join(str(c) for c in cnt) + " |")

    text = "\n".join(md) + "\n"
    with open(os.path.join(a.out, "collapse.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
