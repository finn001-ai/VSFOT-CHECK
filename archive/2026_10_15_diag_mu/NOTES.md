# μ_p 估计器诊断：无标签能收回多少 oracle 上界（2026-10-15）

## 概要

**为什么做**
消融和诊断都表明，OT 模块真正起作用的是类别边际 μ_p，oracle μ_p（本批真实类分布）比 CLIP 高 7–16 个百分点。
如果有不用标签的 μ_p 估计方法能收回其中一部分，这就是一个可以改进方法的方向。

**做了什么**
`bash scripts/run_diag_mu.sh` → `tools/diag_mu.py`，用 `process/DiagFix/` 的 6 个快照。
每批用同一份 CLIP 概率、不改动的 `getGamma`，只换 μ_p：bank（现用）、true_batch（oracle）、uniform、clip_now、
model、clip_model_avg、sharp_T0.5 / T0.25（锐化后取均值）、present_tau0.3 / 0.5（按批内最大概率剔除类）、
em_rowsum1 / 3（健全性检查）、em_mix1 / 3（混合比例 EM）、uot_m0.1 / 1.0（类别边际 KL 松弛的半松弛 OT）。
指标：恢复比例 = (acc_变体 − acc_bank) / (acc_true_batch − acc_bank)。

**结论**

| | CLIP argmax | 最好的无标签方法 | 比 CLIP 高（百分点） | 恢复比例 |
|---|---|---|---|---|
| Ar→Cl p000 | 63.64 | 63.69（uot_m0.1） | +0.05 | 10% |
| Ar→Cl p025 | 72.93 | 72.99（uot_m0.1） | +0.06 | 7% |
| Ar→Cl p050 | 73.58 | 74.15（model） | +0.57 | 6% |
| Pr→Re p000 | 87.72 | 87.71（sharp_T0.25） | −0.01 | 8% |
| Pr→Re p025 | 90.78 | 91.17（clip_model_avg） | +0.39 | 4% |
| Pr→Re p050 | 91.07 | 91.17（present_tau0.3） | +0.10 | 2% |

- 最好的方法（锐化 μ_p、半松弛 OT）只是把计划从略低于 CLIP 拉回到等于 CLIP；「恢复 10%」就是补回 bank 比 CLIP argmax 低的部分。
- 只有模型准确率已超过 CLIP 的阶段（Ar→Cl p050、Pr→Re p025），用模型的 μ_p 才比 CLIP 高 0.4–0.6 个百分点；
  早期模型弱，用模型 μ_p 明显变差。
- present（按阈值剔除类）、em_mix（混合比例 EM）全部为负；em_rowsum ≡ bank，符合预期（Sinkhorn 收敛后行和 ≡ μ）。

**证明了什么**
- μ_p 只要从 CLIP 概率推出，OT 计划的上限就是 CLIP 的 argmax，拿不到 CLIP 以外的信息。
  批内类别约束只有在提供「单看每个样本的 CLIP 看不出的信息」时才有用，而 oracle 用的正是真实标签。
- 「在批内改进无标签 μ_p 估计」这条路，在这些简单方法上走不通。

**没有证明的**
- 只测了简单的估计器；更复杂的方法（例如利用特征聚类、跨批统计）没有测。
- 全数据集级别的 OT（避免批内与全域分布不一致）没有测。
- 静态测量，不代表训练中的效果。

---

原始结果：`results/summary_all.md`（含恢复比例速览和实现记录），逐批数据 `results/<任务>/<阶段>/`。
