"""类先验的无标签估计：特征空间聚类 + 一一匹配（2026-10-23）

run_oracle.sh：真实类先验作均衡目标是目前最大的杠杆（Ar→Pr +1.75、DomainNet R→S +2.24）；
run_clsw_self.sh：用学生自己的预测分布当目标没用（Ar→Cl −0.03）——预测直方图里被 CLIP 混淆吞掉的类本来就少算，冻结它等于冻结混淆。
假设：学生特征里类是可分的（探针 95.6 vs 学生 92.0），在特征空间做 K = C 的聚类，再用一一匹配（匈牙利）把簇对到类，
被吞掉的类会被迫分到自己的簇，簇大小就是不受混淆影响的类先验估计。标签只用于计算 L1 和匹配准确率。

  python tools/diag_prior_cluster.py --snapshot runs/.../snapshot_p025.pt [--device cpu]
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _os.path.join(_ROOT, "tools"))
_os.chdir(_ROOT)
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from diag_repr import extract


def kmeans_cos(x, k, iters=50, restarts=3, seed=0, init=None):
    """余弦 k-means；init 给定时用它做唯一一次初始化（类中心），否则 k-means++ 式随机初始化取惯量最小"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_obj = None, -1e9
    for r in range(1 if init is not None else restarts):
        c = init.clone() if init is not None else x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
        for _ in range(iters):
            a = (x @ c.T).argmax(1)
            c = F.normalize(torch.zeros_like(c).index_add_(0, a, x) + 1e-8 * c, dim=1)
        obj = (x @ c.T).max(1).values.sum().item()
        if obj > best_obj:
            best, best_obj = a, obj
    return best


def match_prior(a, P, C):
    """簇 a ∈ [0, C) 与类一一匹配（最大化簇内平均概率之和），返回 (按类的簇大小先验, 每个样本的匹配类)"""
    M = torch.zeros(C, C, device=P.device).index_add_(0, a, P)          # [簇, 类] 概率质量
    r, c = linear_sum_assignment(-M.cpu().numpy())
    m = torch.empty(C, dtype=torch.long, device=P.device); m[torch.as_tensor(r, device=P.device)] = torch.as_tensor(c, device=P.device)
    lab = m[a]
    return torch.bincount(lab, minlength=C).float() / len(lab), lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()
    torch.set_num_threads(6)
    print("| 快照 | 特征 | 初始化 | L1 均匀 | L1 学生直方图 | L1 聚类匹配 | 相关 直方图 / 聚类 | 学生 acc | 匹配标签 acc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in cli.snapshot:
        snap, feats, q_m, q_v, y = extract(path, cli.device)
        C = q_m.size(1); N = len(y)
        t = torch.bincount(y, minlength=C).float() / N
        L1 = lambda q: (q / q.sum() - t).abs().sum().item()
        corr = lambda q: np.corrcoef((q / q.sum()).cpu().numpy(), t.cpu().numpy())[0, 1]
        hs = torch.bincount(q_m.argmax(1), minlength=C).float()
        P = (q_m + q_v) / 2
        name = path.replace("runs/", "").replace("/snapshot", "")
        for f in ("bottleneck", "pool2048", "clip"):
            x = F.normalize(feats[f], dim=1)
            for mode in ("proto", "random"):
                if mode == "proto":     # 用学生预测（概率加权）的类中心初始化：簇和类天然对齐，被吞掉的类中心从它自己的少量样本出发
                    init = F.normalize(P.T @ x, dim=1)
                    a = kmeans_cos(x, C, init=init)
                else:
                    a = kmeans_cos(x, C)
                prior, lab = match_prior(a, P, C)
                print(f"| {name} | {f} | {mode} | {L1(torch.ones(C, device=t.device)):.3f} | {L1(hs):.3f} | {L1(prior):.3f} | "
                      f"{corr(hs):.2f} / {corr(prior):.2f} | {(q_m.argmax(1) == y).float().mean()*100:.2f} | {(lab == y).float().mean()*100:.2f} |")
