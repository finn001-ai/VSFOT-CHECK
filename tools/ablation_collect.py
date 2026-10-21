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
            "dyn": os.path.join(run_dir, "dynamics.csv") if os.path.isfile(os.path.join(run_dir, "dynamics.csv")) else None,
        })
        dst = os.path.join(a.out, "logs", mode, os.path.basename(run_dir))
        os.makedirs(dst, exist_ok=True)
        for name in ("run_log.txt", "argsInfo.txt", "ModellineAcc.txt", ".done", "dynamics.csv", "samples.npz"):
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
    # ---------------- 跨任务对照：每个模式一行，缺的任务记为 -；附 COSMO（论文表 1，ViT-B/32）作参照 ----------------
    COSMO = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Ar2Re": 92.0, "Cl2Ar": 85.7, "Cl2Pr": 92.4, "Cl2Re": 91.9,
             "Pr2Ar": 84.3, "Pr2Cl": 76.1, "Pr2Re": 91.7, "Re2Ar": 85.6, "Re2Cl": 75.5, "Re2Pr": 92.7}
    if len(tasks) > 1:
        all_modes = sorted({m for _, m in group})
        md += [f"## 跨任务对照（最后 5 次评估均值，多个种子取均值；平均只对该模式已有的任务求）", "",
               "| 模式 | " + " | ".join(tasks) + " | 任务数 | 平均 | 同任务上 COSMO 平均 |", "|---|" + "---|" * (len(tasks) + 3)]
        for m in all_modes:
            have = [t for t in tasks if last5_mean.get((t, m)) is not None]
            cells = [f"{last5_mean[(t, m)] * 100:.2f}" if t in have else "-" for t in tasks]
            avg = statistics.mean(last5_mean[(t, m)] for t in have) * 100
            cos = [COSMO[t] for t in have if t in COSMO]
            md.append(f"| {m} | " + " | ".join(cells) + f" | {len(have)} | {avg:.2f} | "
                      + (f"{statistics.mean(cos):.2f}" if len(cos) == len(have) else "-") + " |")
        if all(t in COSMO for t in tasks):
            md.append("| COSMO（参照） | " + " | ".join(f"{COSMO[t]:.1f}" for t in tasks) + f" | {len(tasks)} | "
                      f"{statistics.mean(COSMO[t] for t in tasks):.2f} | - |")
        md.append("")

    md += ["## 分解（最后 5 次评估均值之差，百分点）", "",
           "| 任务 | 锐化（sharp − soft） | 边际投影（ot − sharp） | OT 总增益（ot − soft） | ot − hard |",
           "|---|---|---|---|---|"]
    md += [f"| {t} | {diff(t, 'clip_sharp', 'clip_soft')} | {diff(t, 'ot', 'clip_sharp')} | "
           f"{diff(t, 'ot', 'clip_soft')} | {diff(t, 'ot', 'clip_hard')} |" for t in tasks]
    md.append("")

    # ---------------- 训练动态（--log_dynamics 的组）----------------
    cols = [("student_acc", "学生", True), ("teacher_acc", "老师", True), ("agree", "一致率", True),
            ("either_acc", "任一判对", True), ("both_wrong", "两边都错", False),
            ("student_flip", "学生翻转", True), ("teacher_flip", "老师翻转", True),
            ("pl_gamma_acc", "伪标签(gamma1)", True), ("pl_align_acc", "伪标签(对齐概率)", True),
            ("mgva_keep", "MGVA保留", True), ("mgva_keep_acc", "保留样本学生准确率", True),
            ("student_ent", "学生熵", False), ("teacher_ent", "老师熵", False),
            ("loss_align", "L_align", False), ("loss_ctr", "L_ctr", False), ("loss_iml", "L_iml", False), ("loss_vil", "L_vil", False)]
    for r in [r for r in runs if r["dyn"]]:
        with open(r["dyn"], encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        pick = [rows[i] for i in sorted({0, 1, 2, 3, 5, 8, 11, 15, 20, 25, 30, 35, len(rows) - 1}) if i < len(rows)]
        md += [f"## 训练动态：{r['mode']} / {r['task']}_s{r['seed']}", "",
               "百分比列为 %；翻转 = 和上一次评估相比改变预测的样本比例；伪标签准确率只在训练批上统计（标签只用于记录）。", "",
               "| iter | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (1 + len(cols))]
        for row in pick:
            cells = []
            for k, _, is_pct in cols:
                v = row.get(k, "")
                try:
                    x = float(v)
                    cells.append("-" if x != x else (f"{x * 100:.2f}" if is_pct else (f"{int(x)}" if k == "both_wrong" else f"{x:.3f}")))
                except ValueError:
                    cells.append("-")
            md.append(f"| {row['iter']} | " + " | ".join(cells) + " |")
        md.append("")

    text = "\n".join(md)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    zip_path = shutil.make_archive(a.out.rstrip("/"), "zip", root_dir=a.out)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
