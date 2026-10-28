"""
门控信息诊断（读 --log_rich 的 rich.npz / rich_feat_*.npz，不训练）：不用标签判断「学生和老师谁更准」

硬预测 / 熵上的信号（archive/2026_10_17_collapse、tools/diag_ds.py）都在 Ar2Cl 上判反；这里用完整概率、翻转视图和两边特征。
候选量（都是「学生得分 − 老师得分」，> 0 表示判学生更准；标签只用于评估）：
  learn    可学性（主判据）：在某个特征空间上用 2 折交叉拟合的 softmax 回归去学一方的预测类，留出折上与该方预测一致的比例；
           学生特征、老师特征两个空间各算一次取平均（两边各自的空间都偏向自己，对称后抵消）。直觉：标签越对，在好的表示里越自洽。
  knn      近邻一致：某方预测类与它在特征空间中 10 个近邻的多数预测相同的比例，两个空间取平均
  flip     翻转稳定：原图与水平翻转预测类相同的比例
  flip_kl  翻转后概率的对称 KL（取负，越稳定越大）
  cross    交叉认可：对方预测类在本方概率中的排名中位数（取负；体现「被对方认可」的程度）
预定判据：learn 在进度 25% 的评估点上 4 个任务符号全对（Ar2Cl、Pr2Cl 应 > 0；Cl2Ar、Re2Ar 应 < 0），
         且在有特征的评估点上与真实「学生 − 老师」符号一致 ≥ 80%；否则不进训练。其他量只作参考，不据此挑选。

用法：python tools/diag_rich.py runs/GateRich/cap02_tstrong/{Ar2Cl,Pr2Cl,Cl2Ar,Re2Ar}_s42
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def learnability(feat, lab, K, steps=300, seed=0):
    """2 折交叉拟合：一折上训练 softmax 回归学 lab，另一折上预测，返回留出预测与 lab 一致的比例"""
    g = torch.Generator().manual_seed(seed)
    N = len(lab)
    perm = torch.randperm(N, generator=g)
    folds = [perm[: N // 2], perm[N // 2:]]
    X = torch.tensor(feat, dtype=torch.float32, device=DEV)
    y = torch.tensor(lab, dtype=torch.long, device=DEV)
    hit = 0
    for a, b in ((0, 1), (1, 0)):
        tr, te = folds[a].to(DEV), folds[b].to(DEV)
        W = torch.zeros(X.shape[1], K, device=DEV, requires_grad=True)
        bias = torch.zeros(K, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([W, bias], lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(10 * X[tr] @ W + bias, y[tr]).backward()
            opt.step()
        with torch.no_grad():
            hit += ((10 * X[te] @ W + bias).argmax(1) == y[te]).sum().item()
    return hit / N


def knn_consistency(feat, lab, k=10):
    X = F.normalize(torch.tensor(feat, dtype=torch.float32, device=DEV), dim=1)
    y = torch.tensor(lab, dtype=torch.long, device=DEV)
    sim = X @ X.T
    sim.fill_diagonal_(-2)
    nb = sim.topk(k, dim=1).indices
    maj = torch.mode(y[nb], dim=1).values
    return (maj == y).float().mean().item()


def sym_kl(p, q):
    p, q = p.astype(np.float64) + 1e-6, q.astype(np.float64) + 1e-6
    return 0.5 * ((p * np.log(p / q)).sum(1) + (q * np.log(q / p)).sum(1)).mean()


def rank_of(prob, lab):
    """lab 在 prob 每行中的名次（0 = 第一）"""
    return (prob > prob[np.arange(len(lab)), lab][:, None]).sum(1)


def analyse(d):
    z = np.load(os.path.join(d, "rich.npz"))
    y, iters = z["label"], z["iter"]
    sp, tp, spf, tpf = z["s_prob"].astype(np.float32), z["t_prob"].astype(np.float32), z["s_prob_flip"], z["t_prob_flip"]
    E, K = len(iters), sp.shape[2]
    feats = {int(os.path.basename(f)[10:15]): f for f in glob.glob(os.path.join(d, "rich_feat_*.npz"))}
    task = os.path.basename(d.rstrip("/"))
    print(f"\n### {task}（N = {len(y)}，{E - 1} 次评估）\n")
    print("| 评估 | 进度 | 真实 学生 − 老师 | learn | knn | flip | flip_kl | cross |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for e in range(1, E):
        s, t = sp[e].argmax(1), tp[e].argmax(1)
        true_d = (s == y).mean() - (t == y).mean()
        flip = (s == spf[e].argmax(1)).mean() - (t == tpf[e].argmax(1)).mean()
        fkl = -(sym_kl(sp[e], spf[e]) - sym_kl(tp[e], tpf[e]))
        cross = -(np.median(rank_of(sp[e], t)) - np.median(rank_of(tp[e], s)))
        learn = knn = np.nan
        if int(iters[e]) in feats:
            f = np.load(feats[int(iters[e])])
            sf, tf = f["s_feat"].astype(np.float32), f["t_feat"].astype(np.float32)
            learn = np.mean([learnability(X, s, K) - learnability(X, t, K) for X in (sf, tf)])
            knn = np.mean([knn_consistency(X, s) - knn_consistency(X, t) for X in (sf, tf)])
        rows.append((e, true_d, learn, knn, flip, fkl, cross))
        g = lambda v: "-" if np.isnan(v) else f"{v * 100:+.2f}"
        print(f"| {e} | {e / (E - 1):.2f} | {true_d * 100:+.2f} | {g(learn)} | {g(knn)} | {flip * 100:+.2f} | {fkl:+.4f} | {cross:+.1f} |")
    R = np.array(rows)
    print("\n符号与真实一致的比例（只算有值的评估点）：", end=" ")
    for j, name in enumerate(["learn", "knn", "flip", "flip_kl", "cross"], start=2):
        m = ~np.isnan(R[:, j]) & (R[:, 1] != 0)
        print(f"{name} {(np.sign(R[m, j]) == np.sign(R[m, 1])).mean():.2f}（{m.sum()} 点）", end="；")
    print()
    m = ~np.isnan(R[:, 2])
    e25 = R[m][np.argmin(np.abs(R[m, 0] / (E - 1) - 0.25))]
    print(f"进度 25% 附近（评估 {int(e25[0])}）learn {e25[2] * 100:+.2f}，真实 {e25[1] * 100:+.2f}")


if __name__ == "__main__":
    print("## 门控信息诊断（单位：百分点；> 0 = 判学生更准）")
    for d in _sys.argv[1:]:
        if os.path.exists(os.path.join(d, "rich.npz")):
            analyse(d)
        else:
            print(f"\n{d}：没有 rich.npz")
