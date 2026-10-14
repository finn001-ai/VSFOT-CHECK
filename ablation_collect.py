"""
汇总 OT 消融实验的结果，并打包成 zip

从 process/Ablation/<模式>/<任务>/run_log.txt 中读取每次评估的准确率

用法：
  python ablation_collect.py --process process/Ablation --out ablation_results/2026_10_15

输出：
  <out>/summary.md     每个任务一张表：各模式的最终 / 最佳准确率，以及训练曲线
  <out>/curves.csv     所有评估点（模式, 任务, iter, acc）
  <out>/logs/          各次训练的日志
  <out>.zip
"""
import os
import re
import csv
import glob
import shutil
import argparse

# main.py 每次评估打印的行：... Iter:  680/2720; Now Acc: 0.6694; Best Acc: 0.6724
PATTERN = re.compile(r"Iter:\s*(\d+)\s*/\s*(\d+); Now Acc: ([0-9.]+); Best Acc: ([0-9.]+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, default="process/Ablation")
    parser.add_argument("--out", type=str, required=True)
    a = parser.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = {}            # (task, mode) -> [(iter, acc, best)]
    max_iters = {}
    for log in sorted(glob.glob(os.path.join(a.process, "*", "*", "run_log.txt"))):
        task_dir = os.path.dirname(log)
        mode, task = os.path.basename(os.path.dirname(task_dir)), os.path.basename(task_dir)
        pts = []
        with open(log, encoding="utf-8") as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    pts.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
                    max_iters[(task, mode)] = int(m.group(2))
        runs[(task, mode)] = pts
        dst = os.path.join(a.out, "logs", mode, task)
        os.makedirs(dst, exist_ok=True)
        for name in ("run_log.txt", "argsInfo.txt", "ModellineAcc.txt", ".done"):
            if os.path.isfile(os.path.join(task_dir, name)):
                shutil.copy(os.path.join(task_dir, name), os.path.join(dst, name))

    if not runs:
        print(f"[错误] {a.process} 下没有找到训练日志"); return

    with open(os.path.join(a.out, "curves.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["mode", "task", "iter", "acc"])
        for (task, mode), pts in sorted(runs.items()):
            for it, acc, _ in pts:
                wr.writerow([mode, task, it, acc])

    md = []
    for task in sorted({t for t, _ in runs}):
        modes = [m for m in ("ot", "clip_soft", "clip_hard") if (task, m) in runs] + \
                sorted(m for t, m in runs if t == task and m not in ("ot", "clip_soft", "clip_hard"))
        md += [f"## {task}", "", "| 模式 | 最终准确率 | 最佳准确率 | 最后 5 次评估均值 | 评估次数 | 是否跑完 |", "|---|---|---|---|---|---|"]
        for m in modes:
            pts = runs[(task, m)]
            if not pts:
                md.append(f"| {m} | - | - | - | 0 | 否 |"); continue
            last5 = [p[1] for p in pts[-5:]]
            done = os.path.isfile(os.path.join(a.process, m, task, ".done"))
            md.append(f"| {m} | {pts[-1][1]:.4f} | {pts[-1][2]:.4f} | {sum(last5) / len(last5):.4f} | {len(pts)} | {'是' if done else '否'} |")
        # 训练曲线：每 5 次评估取一个点，方便对比
        md += ["", "训练曲线（iter: acc）：", ""]
        for m in modes:
            pts = runs[(task, m)]
            md.append(f"- {m}: " + "  ".join(f"{it}:{acc:.4f}" for it, acc, _ in pts[::5] + pts[-1:]))
        md.append("")
    text = "\n".join(md)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    zip_path = shutil.make_archive(a.out.rstrip("/"), "zip", root_dir=a.out)
    print(f"已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
