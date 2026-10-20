"""
--clsw_mode prior 的前提检查：CLIP 零样本平均概率 μ̂ 比均匀分布更接近目标域真实类分布吗（读取训练前快照的 vil_bank，不前向）

vil_bank 在 MemoryBank.initialize 里用原始 CLIP 零样本概率初始化，p000 快照存的就是它；标签只用于检查。
报告每个目标域：μ̂ / 零样本预测计数 / 均匀 与真实分布的 L1 距离（0 = 一致，最大 2），以及最多 / 最少类之比。

用法：python tools/check_prior.py runs/H1Probe/p000/Ar2Pr_s42/snapshot_p000.pt [更多快照 ...]
"""
import sys
import torch

print("| 快照 | 真实 最多/最少 | μ̂ 最多/最少 | L1：μ̂（零样本平均概率） | L1：零样本预测计数 | L1：均匀 |")
print("|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    snap = torch.load(path, map_location="cpu", weights_only=False)
    vb = snap["vil_bank"].float()
    C = vb.shape[1]
    y = torch.tensor([int(item[1]) for item in snap["imgs"]])
    true = torch.bincount(y, minlength=C).float()
    true = true / true.sum()
    mu = vb.mean(0)
    cnt = torch.bincount(vb.argmax(1), minlength=C).float()
    cnt = cnt / cnt.sum()
    uni = torch.full((C,), 1.0 / C)
    l1 = lambda m: (m - true).abs().sum().item()
    name = path.split("/")[-2]
    print(f"| {name} | {(true.max() / true.min()).item():.1f} | {(mu.max() / mu.min()).item():.1f} | "
          f"{l1(mu):.3f} | {l1(cnt):.3f} | {l1(uni):.3f} |")
