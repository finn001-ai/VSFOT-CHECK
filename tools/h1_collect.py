"""
H1 汇总：CLIP 强的目标域上，限制我们的是不是「学生没学到 CLIP 的表示」（读取 tools/diag_repr.py 的 result.json）

目录结构（scripts/run_h1.sh 生成）：<root>/<任务>/<阶段>/result.json，阶段 = p000（训练前）或 p100（cap02 训练结束）
p000：学生 = 源模型，老师 = 刚插入 adapter 的 CLIP（adapter 初始化很小，≈ 原始 CLIP，准确率应接近零样本）
探针 = 5 折交叉验证的线性逻辑回归（用标签，只作上限参考）

用法：python tools/h1_collect.py --root runs/h1_results/xxx
汇总写到 <root>/summary.md 并完整打印（贴这一段即可）
"""
import os
import glob
import json
import argparse
import statistics

TASKS = ["Ar2Cl", "Pr2Cl", "Re2Cl", "Cl2Ar", "Pr2Ar", "Re2Ar", "Ar2Pr", "Cl2Pr", "Re2Pr", "Ar2Re", "Cl2Re", "Pr2Re"]
# cap02，seed 42，最后 5 次评估均值（archive/2026_10_18_weight）
CAP02 = {"Ar2Cl": 77.00, "Ar2Pr": 91.91, "Ar2Re": 91.31, "Cl2Ar": 83.65, "Cl2Pr": 92.06, "Cl2Re": 91.48,
         "Pr2Ar": 84.35, "Pr2Cl": 77.21, "Pr2Re": 91.48, "Re2Ar": 84.27, "Re2Cl": 77.04, "Re2Pr": 92.81}
COSMO = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Ar2Re": 92.0, "Cl2Ar": 85.7, "Cl2Pr": 92.4, "Cl2Re": 91.9,
         "Pr2Ar": 84.3, "Pr2Cl": 76.1, "Pr2Re": 91.7, "Re2Ar": 85.6, "Re2Cl": 75.5, "Re2Pr": 92.7}
TARGET = {"Cl": "Clipart", "Ar": "Art", "Pr": "Product", "Re": "Real World"}


def load(root, task, stage):
    f = os.path.join(root, task, stage, "result.json")
    return json.load(open(f, encoding="utf-8")) if os.path.isfile(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    p2 = lambda v: "-" if v is None or v != v else f"{v * 100:.2f}"
    probe = lambda r, L: r["layers"][L]["probe"] if r and L in r["layers"] else None
    mut = lambda r, L: r["layers"][L]["lab_mutual"] if r and L in r["layers"] else None

    md = ["# H1：CLIP 强的目标域上，学生有没有学到 CLIP 的表示", "",
          "p000 = 训练前（学生 = 源模型，老师 ≈ 原始 CLIP）；p100 = cap02 训练结束。探针 = 5 折 CV 线性探针（用标签，上限参考）。",
          "判读：p000 的 CLIP 探针明显高于 cap02 / COSMO → CLIP 的表示里有我们没传给学生的信息（H1 成立）；",
          "p100 学生特征探针 ≈ cap02 学生 → 学生表示就是上限；p100 学生探针明显更高 → 信息在学生表示里，是分类器没用上。", "",
          "## 1. 训练前（p000）",  "",
          "| 任务 | 源模型 | CLIP 零样本 | 探针：源模型 pool2048 | 探针：源模型 bottleneck | **探针：CLIP** | 互近邻标签：CLIP | cap02 | COSMO | CLIP 探针 − cap02 | CLIP 探针 − COSMO |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for t in TASKS:
        r = load(a.root, t, "p000")
        if r is None:
            md.append(f"| {t} | - | - | - | - | - | - | {CAP02[t]:.2f} | {COSMO[t]:.1f} | - | - |")
            continue
        cp = probe(r, "clip")
        rows[t] = cp
        md.append(f"| {t} | {p2(r['student'])} | {p2(r['teacher'])} | {p2(probe(r, 'pool2048'))} | {p2(probe(r, 'bottleneck'))} | "
                  f"**{p2(cp)}** | {p2(mut(r, 'clip'))} | {CAP02[t]:.2f} | {COSMO[t]:.1f} | "
                  f"{cp * 100 - CAP02[t]:+.2f} | {cp * 100 - COSMO[t]:+.2f} |")

    md += ["", "## 2. 按目标域平均（p000）", "",
           "| 目标域 | CLIP 零样本 | 探针：CLIP | 探针：源模型 pool2048 | cap02 | COSMO | CLIP 探针 − cap02 | CLIP 探针 − COSMO |",
           "|---|---|---|---|---|---|---|---|"]
    for k, name in TARGET.items():
        ts = [t for t in TASKS if t.endswith(k) and t in rows]
        if not ts:
            continue
        R = [load(a.root, t, "p000") for t in ts]
        m = lambda xs: statistics.mean(xs)
        zs, cp, sp = m([r["teacher"] for r in R]) * 100, m([probe(r, "clip") for r in R]) * 100, m([probe(r, "pool2048") for r in R]) * 100
        c2, co = m([CAP02[t] for t in ts]), m([COSMO[t] for t in ts])
        md.append(f"| {name}（{len(ts)} 个任务） | {zs:.2f} | {cp:.2f} | {sp:.2f} | {c2:.2f} | {co:.2f} | {cp - c2:+.2f} | {cp - co:+.2f} |")

    md += ["", "## 3. cap02 训练结束（p100，有快照的任务）", "",
           "| 任务 | 学生 | 老师 | 探针：学生 pool2048 | 探针：学生 bottleneck | 探针：老师 CLIP（adapter 后） | p000 探针：CLIP | 学生探针 − 学生 |",
           "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        r = load(a.root, t, "p100")
        if r is None:
            continue
        r0 = load(a.root, t, "p000")
        best = max(probe(r, "pool2048"), probe(r, "bottleneck"))
        md.append(f"| {t} | {p2(r['student'])} | {p2(r['teacher'])} | {p2(probe(r, 'pool2048'))} | {p2(probe(r, 'bottleneck'))} | "
                  f"{p2(probe(r, 'clip'))} | {p2(probe(r0, 'clip'))} | {(best - r['student']) * 100:+.2f} |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(a.root, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print(text)


if __name__ == "__main__":
    main()
