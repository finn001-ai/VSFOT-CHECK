"""
汇总 OT 消融实验的结果，并打包成 zip

从 runs/<训练目录>/<模式>/<任务>_s<种子>/run_log.txt 中读取每次评估的准确率和训练耗时；
同一任务、同一模式的多个种子取平均。

用法（一般由 scripts/run_ablation.sh 调用）：
  python tools/ablation_collect.py --process runs/AblationFix --out runs/ablation_results/2026_10_16

输出：
  <out>/summary.md     每个任务一张表 + 训练曲线 + 分解表
  <out>/runs.csv       每次训练一行（模式, 任务, 种子, 最终 / 最佳 / 最后 5 次均值, 耗时）
  <out>/curves.csv     所有评估点（模式, 任务, 种子, iter, acc）
  <out>/logs/          各次训练的日志
  <out>.zip
"""

import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
import re
import csv
import glob
import shutil
import argparse
import statistics

ORDER = ("ot", "clip_soft", "clip_sharp", "clip_hard")   # 表格中模式的显示顺序

# main.py 每次评估写入的行：... Iter:  680/2720; Now Acc: 0.6694; Best Acc: 0.6724
PATTERN = re.compile(r"Iter:\s*(\d+)\s*/\s*(\d+); Now Acc: ([0-9.]+); Best Acc: ([0-9.]+)")
# main.py 结束时写入的行：【总耗时】0:31:05
ELAPSED = re.compile(r"【总耗时】(\d+):(\d+):(\d+)")


def mean_std(vals):
    """均值和标准差（单个种子时标准差为 None）"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def fmt_pct(m, s=None):
    if m is None:
        return "-"
    return f"{m * 100:.2f}" + (f" ± {s * 100:.2f}" if s is not None else "")


def fmt_min(sec):
    return "-" if sec is None else f"{sec / 60:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, required=True, help="训练输出目录，如 runs/AblationFix")
    parser.add_argument("--out", type=str, required=True)
    a = parser.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------- 读取每次训练 ----------------
    runs = []       # 每次训练一个 dict
    for log in sorted(glob.glob(os.path.join(a.process, "*", "*", "run_log.txt"))):
        run_dir = os.path.dirname(log)
        mode = os.path.basename(os.path.dirname(run_dir))
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(run_dir))
        task, seed = (m.group(1), int(m.group(2))) if m else (os.path.basename(run_dir), None)
        pts, elapsed = [], None
        with open(log, encoding="utf-8") as f:
            for line in f:
                p = PATTERN.search(line)
                if p:
                    pts.append((int(p.group(1)), float(p.group(3)), float(p.group(4))))
                e = ELAPSED.search(line)
                if e:
                    elapsed = int(e.group(1)) * 3600 + int(e.group(2)) * 60 + int(e.group(3))
        runs.append({
            "mode": mode, "task": task, "seed": seed, "pts": pts, "elapsed": elapsed,
            "done": os.path.isfile(os.path.join(run_dir, ".done")),
            "final": pts[-1][1] if pts else None,
            "best": pts[-1][2] if pts else None,
            "last5": statistics.mean(p[1] for p in pts[-5:]) if pts else None,
        })
        dst = os.path.join(a.out, "logs", mode, os.path.basename(run_dir))
        os.makedirs(dst, exist_ok=True)
        for name in ("run_log.txt", "argsInfo.txt", "ModellineAcc.txt", ".done"):
            if os.path.isfile(os.path.join(run_dir, name)):
                shutil.copy(os.path.join(run_dir, name), os.path.join(dst, name))

    if not runs:
        print(f"[错误] {a.process} 下没有找到训练日志"); return

    # ---------------- 明细 CSV ----------------
    with open(os.path.join(a.out, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["mode", "task", "seed", "final", "best", "last5", "elapsed_sec", "n_eval", "done"])
        for r in runs:
            wr.writerow([r["mode"], r["task"], r["seed"], r["final"], r["best"], r["last5"], r["elapsed"], len(r["pts"]), r["done"]])
    with open(os.path.join(a.out, "curves.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["mode", "task", "seed", "iter", "acc"])
        for r in runs:
            for it, acc, _ in r["pts"]:
                wr.writerow([r["mode"], r["task"], r["seed"], it, acc])

    # ---------------- 按（任务, 模式）汇总多个种子 ----------------
    group = {}
    for r in runs:
        group.setdefault((r["task"], r["mode"]), []).append(r)
    tasks = sorted({t for t, _ in group})

    md = ["说明：准确率为 %，多个种子时显示「均值 ± 标准差」。比较请看「最后 5 次评估均值」，"
          "「最佳准确率」是在测试集上挑出来的，有选择偏差。", ""]
    last5_mean = {}
    for task in tasks:
        modes = [m for m in ORDER if (task, m) in group] + sorted(m for t, m in group if t == task and m not in ORDER)
        md += [f"## {task}", "",
               "| 模式 | 种子 | 最后 5 次评估均值 | 相对 ot（百分点） | 最终准确率 | 最佳准确率 | 平均耗时（分钟） | 全部跑完 |",
               "|---|---|---|---|---|---|---|---|"]
        ot_l5 = mean_std([r["last5"] for r in group.get((task, "ot"), [])])[0]
        for m in modes:
            rs = group[(task, m)]
            l5 = mean_std([r["last5"] for r in rs])
            last5_mean[(task, m)] = l5[0]
            vs_ot = f"{(l5[0] - ot_l5) * 100:+.2f}" if (l5[0] is not None and ot_l5 is not None and m != "ot") else "-"
            el = mean_std([r["elapsed"] for r in rs])[0]
            seeds = ",".join(str(r["seed"]) for r in rs)
            md.append(f"| {m} | {seeds} | {fmt_pct(*l5)} | {vs_ot} | {fmt_pct(*mean_std([r['final'] for r in rs]))} | "
                      f"{fmt_pct(*mean_std([r['best'] for r in rs]))} | {fmt_min(el)} | {'是' if all(r['done'] for r in rs) else '否'} |")
        # 训练曲线：多个种子时取各评估点的均值；每 5 次评估取一个点
        md += ["", "训练曲线（iter: acc，多个种子取均值）：", ""]
        for m in modes:
            rs = [r for r in group[(task, m)] if r["pts"]]
            if not rs:
                continue
            n = min(len(r["pts"]) for r in rs)
            curve = [(rs[0]["pts"][i][0], statistics.mean(r["pts"][i][1] for r in rs)) for i in range(n)]
            md.append(f"- {m}: " + "  ".join(f"{it}:{acc * 100:.2f}" for it, acc in curve[::5] + curve[-1:]))
        md.append("")

    # ---------------- 分解：锐化 / 边际投影各自的作用 ----------------
    def diff(task, x, y):
        vx, vy = last5_mean.get((task, x)), last5_mean.get((task, y))
        return f"{(vx - vy) * 100:+.2f}" if vx is not None and vy is not None else "-"
    md += ["## 分解（最后 5 次评估均值之差，百分点）", "",
           "| 任务 | 锐化（sharp − soft） | 边际投影（ot − sharp） | OT 总增益（ot − soft） | ot − hard |",
           "|---|---|---|---|---|"]
    md += [f"| {t} | {diff(t, 'clip_sharp', 'clip_soft')} | {diff(t, 'ot', 'clip_sharp')} | "
           f"{diff(t, 'ot', 'clip_soft')} | {diff(t, 'ot', 'clip_hard')} |" for t in tasks]
    md.append("")

    text = "\n".join(md)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    zip_path = shutil.make_archive(a.out.rstrip("/"), "zip", root_dir=a.out)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
