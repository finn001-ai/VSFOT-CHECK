"""
潜在类别模型（Dawid–Skene）门控诊断：不用标签估计「学生和老师谁更准」（整体 / 按类），读 --log_samples 的 samples.npz，不训练

以前的单一信号（熵、稳定性、源模型 / 零样本投票，archive/2026_10_17_collapse）都失败；这里把多路信息放进一个联合模型：
  标注者 = 当前学生、当前老师、源模型（第 0 次评估的学生）、零样本 CLIP（第 0 次评估的老师）、最近 W 次评估的学生 / 老师，
  可选按熵分档：同一来源的高置信 / 低置信一半当成两个标注者（另一半记为缺失），各自有自己的准确率。
  每个标注者的模型：P(标 k' | 真类 k) = a[k]（k' = k），否则 (1 − a[k]) · e[k']（e = 该标注者的全局错标分布）。EM 估计 a、e、类先验 π 和后验。
估计量（标签只用于评估）：
  整体：学生 / 老师的估计准确率 Σ_k π_k a[k]，差的符号与真实「学生 − 老师」比较
  按类：每类挑估计准确率更高的一方，组合后的准确率（对照：总听学生 / 总听老师 / 按类 oracle）
  后验：DS 后验 argmax 的准确率（对照 max(学生, 老师)）
预定判据：整体差的符号在评估点上与真实一致 ≥ 80%，且 Ar2Cl（学生领先）与 Cl2Ar（老师领先）方向都对；否则 DS 门控不做。

用法：python tools/diag_ds.py runs/WeightSched/both_cap02_dyn/{Ar2Cl,Cl2Ar,Re2Ar}_s42/samples.npz
"""

import sys
import numpy as np

VARIANTS = {          # 名称：(历史窗口 W，是否按熵分档，是否加源模型 / 零样本)
    "ds2":      (0, False, False),     # 只有当前学生 / 老师（理论上不可辨识，对照）
    "ds4":      (0, False, True),
    "ds4_bin":  (0, True, True),
    "ds_w3":    (3, False, True),
    "ds_w3bin": (3, True, True),
}


def ds_em(L, K, iters=50, alpha=1.0):
    """L: [N, J] 标注（−1 = 缺失）。返回后验 T [N, K]、每个标注者的类准确率 a [J, K]、类先验 pi [K]"""
    N, J = L.shape
    T = np.full((N, K), 1e-3)
    for j in range(J):                                    # 初始化：多数票
        m = L[:, j] >= 0
        T[np.where(m)[0], L[m, j]] += 1
    T /= T.sum(1, keepdims=True)
    for _ in range(iters):
        pi = T.mean(0) + 1e-6; pi /= pi.sum()
        a, e = np.zeros((J, K)), np.zeros((J, K))
        logP = np.tile(np.log(pi), (N, 1))
        for j in range(J):
            m = L[:, j] >= 0
            Tm, lj = T[m], L[m, j]
            hit = Tm[np.arange(len(lj)), lj]              # 真类 = 标注类的后验质量
            onehot = np.zeros((len(lj), K)); onehot[np.arange(len(lj)), lj] = 1
            a[j] = (onehot * Tm).sum(0) + alpha
            a[j] /= Tm.sum(0) + 2 * alpha
            e[j] = (onehot * (1 - hit)[:, None]).sum(0) + alpha
            e[j] /= e[j].sum()
            # log P(l | k)：k = l 时 log a[k]，否则 log (1 − a[k]) + log e[l]
            lp = np.log(1 - a[j])[None, :] + np.log(e[j][lj])[:, None]
            lp[np.arange(len(lj)), lj] = np.log(a[j][lj])
            logP[m] += lp
        logP -= logP.max(1, keepdims=True)
        T = np.exp(logP); T /= T.sum(1, keepdims=True)
    return T, a, pi


def build(z, e, W, binned, ref):
    """第 e 次评估的标注矩阵；返回 L 和「当前学生 / 当前老师」各自对应的列（分档时为两列）"""
    s, t, se, te = z["s_pred"], z["t_pred"], z["s_ent"], z["t_ent"]
    cols, idx = [], {"S": [], "T": []}

    def add(pred, ent, key=None):
        if binned:
            hi = ent <= np.median(ent)
            for m in (hi, ~hi):
                if key: idx[key].append(len(cols))
                cols.append(np.where(m, pred, -1))
        else:
            if key: idx[key].append(len(cols))
            cols.append(pred)

    add(s[e], se[e], "S"); add(t[e], te[e], "T")
    for w in range(1, W + 1):
        if e - w >= 1:
            add(s[e - w], se[e - w]); add(t[e - w], te[e - w])
    if ref:
        add(s[0], se[0]); add(t[0], te[0])
    return np.stack(cols, 1), idx


def est_acc(L, a, pi, cols):
    """标注者组（分档时两列，各占一半样本）的估计整体准确率 Σ_k π_k a[k]，按覆盖样本数加权"""
    w = np.array([(L[:, c] >= 0).mean() for c in cols])
    return sum(wi * (pi * a[c]).sum() for wi, c in zip(w, cols)) / w.sum()


def analyse(path):
    z = np.load(path)
    y, s, t = z["label"], z["s_pred"], z["t_pred"]
    K, E = int(max(y.max(), s.max(), t.max())) + 1, len(z["iter"])
    task = path.split("/")[-2]
    true_d = np.array([(s[e] == y).mean() - (t[e] == y).mean() for e in range(E)])
    print(f"\n### {task}（N = {len(y)}，{E - 1} 次评估；真实「学生 > 老师」比例 {(true_d[1:] > 0).mean():.2f}）\n")
    print("| 变体 | 整体符号一致 | 进度 25% 处 估计差 / 真实差 | 后半段 估计差均值 / 真实差均值 | 后验准确率 − max(S,T) | 按类挑 − max(S,T) | 按类 oracle − max(S,T) |")
    print("|---|---|---|---|---|---|---|")
    e25 = int(round(0.25 * (E - 1)))
    for name, (W, binned, ref) in VARIANTS.items():
        est_d, post_gain, cls_gain, orc_gain = np.zeros(E), [], [], []
        for e in range(1, E):
            L, idx = build(z, e, W, binned, ref)
            T, a, pi = ds_em(L, K)
            est_d[e] = est_acc(L, a, pi, idx["S"]) - est_acc(L, a, pi, idx["T"])
            best = max((s[e] == y).mean(), (t[e] == y).mean())
            post_gain.append((T.argmax(1) == y).mean() - best)
            aS = np.mean([a[c] for c in idx["S"]], 0); aT = np.mean([a[c] for c in idx["T"]], 0)
            # 按类挑：以「学生预测的类」为单位，该类上估计准确率更高的一方说了算
            pick_s = aS[s[e]] >= aT[s[e]]
            comb = np.where(pick_s, s[e], t[e])
            cls_gain.append((comb == y).mean() - best)
            sc = np.bincount(s[e], weights=(s[e] == y), minlength=K); tc = np.bincount(s[e], weights=(t[e] == y), minlength=K)
            orc = np.where((sc >= tc)[s[e]], s[e], t[e])
            orc_gain.append((orc == y).mean() - best)
        agree = (np.sign(est_d[1:]) == np.sign(true_d[1:])).mean()
        h = slice(E // 2, E)
        print(f"| {name} | {agree:.2f} | {est_d[e25] * 100:+.2f} / {true_d[e25] * 100:+.2f} | "
              f"{est_d[h].mean() * 100:+.2f} / {true_d[h].mean() * 100:+.2f} | {np.mean(post_gain) * 100:+.2f} | "
              f"{np.mean(cls_gain) * 100:+.2f} | {np.mean(orc_gain) * 100:+.2f} |")


if __name__ == "__main__":
    print("## Dawid–Skene 门控诊断（单位：百分点；后三列为全部评估点的均值）")
    for p in sys.argv[1:]:
        analyse(p)
