"""分歧当作部分标签的离线可行性检查（2026-10-23）

轨迹诊断：塌缩丢掉的样本在某个时刻至少一方判对过，但按轨迹 / 熵挑一方都救不回来。
假设：不在单个样本上裁决，而是把训练中出现过的候选类都留下（部分标签），交给特征结构跨样本消歧。
做法：冻结特征（零样本 CLIP 图像特征 / 源模型 pool2048）上训练线性头，候选集取自 samples.npz（同一任务的 cap02 训练轨迹）：
  final      最终学生预测（硬标签，= 蒸馏最终学生，对照）
  pair_end   最终学生 ∪ 最终老师
  window_k   训练中任一方连续保持 ≥ k 次评估的类（时间候选集）
损失 = −log Σ_{c∈候选} p_c（PRODEN 式；只用候选集，不用标签）。标签只用于统计覆盖率和准确率。

  python tools/diag_pll.py --snapshot runs/H1Probe/p000/Ar2Cl_s42/snapshot_p000.pt \
      --samples runs/WeightSched/both_cap02_dyn/Ar2Cl_s42/samples.npz
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _os.path.join(_ROOT, "tools"))
_os.chdir(_ROOT)
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from diag_repr import extract


def cand_sets(d, C, k):
    """训练中（不算源模型这一次）学生或老师连续保持 ≥ k 次评估的类 → [N, C] 0/1"""
    N = d["label"].shape[0]
    M = np.zeros((N, C), bool)
    for P in (d["s_pred"][1:], d["t_pred"][1:]):
        run = np.ones(N, int)
        for e in range(len(P)):
            if e > 0:
                run = np.where(P[e] == P[e - 1], run + 1, 1)
            ok = run >= k
            M[np.where(ok)[0], P[e][ok]] = True
    return M


def train_head(x, M, epochs=300, lr=1e-2, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    W = torch.nn.Linear(x.size(1), M.size(1)).to(x.device)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        p = torch.softmax(W(x), dim=1)
        loss = -torch.log((p * M).sum(1) + 1e-8).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W(x).argmax(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--samples", required=True)
    cli = ap.parse_args()
    dev = "cuda:0"
    snap, feats, q_m, q_v, y = extract(cli.snapshot, dev)
    d = np.load(cli.samples)
    C = q_m.size(1)
    assert (torch.from_numpy(d["label"]).to(dev) == y).all(), "samples.npz 与快照的样本顺序不一致"
    yn = y.cpu().numpy()
    fs, ft = d["s_pred"][-1], d["t_pred"][-1]
    print(f"最终学生 {(fs == yn).mean()*100:.2f}  老师 {(ft == yn).mean()*100:.2f}  任一 {((fs == yn) | (ft == yn)).mean()*100:.2f}")
    sets = {"final": np.eye(C, dtype=bool)[fs], "pair_end": np.eye(C, dtype=bool)[fs] | np.eye(C, dtype=bool)[ft]}
    for k in (1, 2, 3, 5, 8):
        sets[f"window_{k}"] = cand_sets(d, C, k)
    print("| 候选集 | 覆盖率 | 平均大小 | 单类占比 | " + " | ".join(f"{f} 线性头" for f in ("clip", "pool2048")) + " |")
    print("|---|---|---|---|---|---|")
    for name, M in sets.items():
        cov = M[np.arange(len(yn)), yn].mean() * 100
        sz = M.sum(1)
        row = f"| {name} | {cov:.2f} | {sz.mean():.2f} | {(sz == 1).mean()*100:.1f}% |"
        Mt = torch.from_numpy(M).float().to(dev)
        for f in ("clip", "pool2048"):
            x = F.normalize(feats[f], dim=1)
            pred = train_head(x, Mt).cpu().numpy()
            row += f" {(pred == yn).mean()*100:.2f} |"
        print(row)
    for f in ("clip", "pool2048"):     # 参考：同一特征上用真实标签训练（全量拟合，非交叉验证）
        x = F.normalize(feats[f], dim=1)
        pred = train_head(x, torch.from_numpy(np.eye(C, dtype=bool)[yn]).float().to(dev)).cpu().numpy()
        print(f"参考：{f} 真实标签拟合 {(pred == yn).mean()*100:.2f}")
