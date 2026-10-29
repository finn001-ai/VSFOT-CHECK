"""
类级边际分歧诊断（读 --log_samples 的 samples.npz，不训练）：能否不用标签提前找出「会被整类吞掉」的类

q1028 原因分析：共识在 →Clipart 上的损失集中在少数易混类整类消失（Ar2Cl Computer 预测数 107 → 0，Pencil / Marker 被 Pen 吞；
Pr2Cl Clipboards 被 Folder 吞），起因是 CLIP 初始偏差（老师 Computer 只预测 10 张，真实 99）在老师侧锁死。
样本级「谁更准」不可辨识（diag_ds / diag_rich 全部判反），这里不判断谁更准，只看两边对「每类有多少张」的估计是否严重分歧：
  分歧 d_k = log((老师预测为 k 的张数 + 1) / (学生预测为 k 的张数 + 1))，|d_k| > log 3 记为分歧类 F（阈值事先定，不调）
  代理组合：学生或老师的预测落在 F 中时用学生的预测，否则用老师的预测（共识训练中老师的预测即融合目标的近似）
评估（标签只用于评估）：分歧类里有没有被吞的类；代理组合相对老师 / 学生的准确率
预定判据：第 1 或 2 次评估时 Ar2Cl 的 F 含 Computer / Pencil / Marker 中至少 2 个、Pr2Cl 的 F 含 Clipboards；
         且在 ctu_nb 共识阶段（第 1–10 次评估）代理组合在 4 个任务上都不低于老师（均值）。否则这条路不进训练。
注意：代理组合只是必要条件筛选，训练中改了目标后动态会变。

用法：python tools/diag_marg.py runs/WeightSched/cap02_tstrong_ctu_nb/{Ar2Cl,Pr2Cl,Cl2Ar,Re2Ar}_s42/samples.npz
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)

import numpy as np

NAMES = [l.strip() for l in open("dataset/OfficeHomeDataset/classname.txt", encoding="utf-8") if l.strip()]
EATEN = {"Ar2Cl": ["Computer", "Pencil", "Marker"], "Pr2Cl": ["Clipboards"]}
THR = np.log(3)


def flagged(s, t, K):
    cs, ct = np.bincount(s, minlength=K), np.bincount(t, minlength=K)
    d = np.log((ct + 1) / (cs + 1))
    return np.where(np.abs(d) > THR)[0], cs, ct


def analyse(path, n_cons=10):
    z = np.load(path)
    y, S, T = z["label"], z["s_pred"], z["t_pred"]
    K, E = len(NAMES), len(z["iter"])
    task = path.split("/")[-2].split("_")[0]
    true_c = np.bincount(y, minlength=K)
    print(f"\n### {task}（N = {len(y)}）\n")
    print("| 评估 | 分歧类（老师张数 / 学生张数 / 真实张数） | 老师 | 学生 | 代理组合 | 组合 − 老师 |")
    print("|---|---|---|---|---|---|")
    gains, hit = [], set()
    for e in range(0, min(E, n_cons + 1)):
        s, t = S[e], T[e]
        F, cs, ct = flagged(s, t, K)
        inF = np.isin(s, F) | np.isin(t, F)
        comb = np.where(inF, s, t)
        at, as_, ac = (t == y).mean(), (s == y).mean(), (comb == y).mean()
        if e >= 1:
            gains.append(ac - at)
        if e in (1, 2):
            hit |= {NAMES[k] for k in F}
        desc = "、".join(f"{NAMES[k]} {ct[k]}/{cs[k]}/{true_c[k]}" for k in F[np.argsort(-np.abs(np.log((ct[F] + 1) / (cs[F] + 1))))]) or "无"
        print(f"| {e} | {desc} | {at * 100:.2f} | {as_ * 100:.2f} | {ac * 100:.2f} | {(ac - at) * 100:+.2f} |")
    msg = f"第 1–{n_cons} 次评估 组合 − 老师 均值 {np.mean(gains) * 100:+.2f}"
    if task in EATEN:
        got = [c for c in EATEN[task] if c in hit]
        msg += f"；第 1 / 2 次评估分歧类中的被吞类：{got or '无'}（应含 {EATEN[task]}）"
    print("\n" + msg)
    return task, np.mean(gains), hit


if __name__ == "__main__":
    print(f"## 类级边际分歧诊断（|log(老师张数+1)/(学生张数+1)| > log 3；单位：%）")
    res = [analyse(p) for p in _sys.argv[1:]]
    ok_hit = all(sum(c in h for c in EATEN[t]) >= (2 if t == "Ar2Cl" else 1) for t, _, h in res if t in EATEN)
    ok_gain = all(g >= 0 for _, g, _ in res)
    print(f"\n判据：被吞类检出 {'过' if ok_hit else '不过'}；代理组合不低于老师 {'过' if ok_gain else '不过'}")
