# 2026-10-28 q1028：OH Cl→Ar 共识析因 + VisDA 30% 老师目标尖锐度 + IM 符号（scripts/run_q1028.sh，seed 42）

## 概要
- 为什么：OH 不加共识的规则只在 Ar→Cl（学生 > 老师 0.95）上定的；Cl→Ar 学生 > 老师只有 0.17，接近 DN / VisDA，从没加过共识。
  另外 GPT 评估指出 MGVA 目标 softmax(10·top-k) 的候选外质量随类别数变（VisDA 9.7% / OH 42% / DN 59%），怀疑共识的收益是「目标变尖」。
- 结论：
  1. **Cl→Ar 加老师侧共识 84.94（cap02_tstrong 83.77，+1.18）**，超过 oracle 先验上界 84.73；COSMO 85.7 仍差 0.76。最后一次 84.96。
     self + 共识 84.31（+0.54）：self 和共识叠加反而不如只加共识。
  2. VisDA 30%：topk1 88.26（+0.20）、topk2 87.99（−0.07），共识 88.56（+0.50）。car：topk1 66.2 vs 共识 71.4 vs 对照 65.5。
     → 共识的收益最多 40% 来自目标变尖，主要来自候选类本身（car / truck 边界）。尖锐度假设基本否定。
  3. IM 正确符号（--imloss -1）87.95（−0.11），噪声内略差，与用户记忆一致。不采用。

| Cl→Ar | 无共识 | 老师侧共识 |
|---|---|---|
| inv | 83.77 | **84.94** |
| self | 84.04 | 84.31 |

## 证明了什么 / 没有证明的
- 支持「共识该不该开取决于学生能不能追上老师，而不是数据集」：OH 内部 Cl→Ar（0.17）+1.18，Ar→Cl（0.95，旧 50% cap02 底座）−1.61。
- 单 seed；Ar→Cl 在当前底座（tstrong，全量）上加共识还没测 → run_ct12.sh 铺开 OH 12 任务，规则：只看 12 任务平均是否 > 86.50，不按任务挑开关。
- 运维：OH 单任务约 24G 显存，不能两路并行；新实例首次运行会下载 CLIP，不能多进程同时下载。

## OH 12 任务 + 老师侧共识（scripts/run_ct12.sh，seed 42，40 epoch，后 5）

12 任务平均：cap02_tstrong 86.50 → + 共识 86.07（最后一次 86.05），COSMO 86.41。预定规则（> 86.50）不通过，OH 维持不加共识。
按目标域：→Ar +0.74（3/3 涨）；→Cl −1.78（Ar2Cl −2.64、Pr2Cl −1.90、Re2Cl −0.80）；→Pr −0.56；→Re −0.12。
与「学生追不上老师时共识有益」一致：→Ar 学生落后老师，→Cl 学生领先。
门控：无标签的逐样本 / 整体判别 10-17 已测失败（archive/2026_10_17_collapse）；只有 ResNet、CLIP 两条来源，Dawid–Skene 不可辨识，需第三个独立视角（待用户定是否在范围内）。

## Dawid–Skene 门控诊断（tools/diag_ds.py，离线，both_cap02_dyn 的 samples.npz：Ar2Cl / Cl2Ar / Re2Ar）
标注者 = 当前学生 / 老师 + 源模型 / 零样本 + 最近 3 次评估，可按熵分档；类条件 one-coin，EM。
预定判据（整体符号一致 ≥ 80%，且 Ar2Cl、Cl2Ar 方向都对）不通过：
- Ar2Cl（真实学生领先 95%）：5 个变体符号一致 0.05–0.33，进度 25% 处全部判「老师更准」（估计 −1.0 ~ −4.3，真实 +2.08）。
- Cl2Ar 0.45–0.72，Re2Ar 0.45–0.70。
- 后验准确率 − max(S,T)：Ar2Cl −0.10 ~ −0.99；Cl2Ar / Re2Ar 最多 +0.29 / +0.48。按类挑全部 ≤ +0.02。
- 按类 oracle（用标签）只比 max(S,T) 高 +1.0：按类在两边硬预测之间挑，上限本来就小。
解读：加了源模型 / 零样本后整体偏向「老师更准」（与 10-17 的 ref_zs 一样），两条谱系的问题没被多来源解决。

## 原因分析 1：共识在 →Clipart 上的损失 = 易混类对上整类被吞（逐类准确率 + 最后一次混淆矩阵，后 5 均值）
- 最差 5 类占变差总量 67–74%（Ar2Cl / Pr2Cl / Re2Cl）；→Ar 上分散得多（58–65%，量也小）。
- Ar2Cl：Computer 60→0（预测数 107→0，67 张进 Monitor；Monitor 69→93），Pencil 57→3（→Pen 70），Marker 44→6（→Pen 38 / Hammer 22）。
- Pr2Cl：Clipboards 78→0（→Folder 29），Pencil 63→8（→Pen 64）。Re2Cl：Computer 59→31，Marker 37→6，Clipboards 78→36。
- Re2Ar 方向相反：Monitor 41→15（→Computer），Computer 46→75，净 +16 样本。VisDA car/truck 也是「吞对方向」。
- 解读：熵加权 PoE + 老师学共识 = 易混类对上赢家通吃的正反馈；吞对方向涨、吞错方向跌，不是「谁更可靠」的问题——解释了逐样本门控为何全部失败。
- 待验证：学生侧有 OT 类边际 + 1/频率权重，类仍能到 0；猜测 μ_p 取自 CLIP 记忆库，老师先杀死类后边际跟着为 0。
  → scripts/run_rich_ct.sh（Ar2Cl / Re2Ar 加共识 + --log_rich），排在 run_rich.sh 之后自动开始。

## 原因分析 2：塌缩不是 PoE 的静态偏差，而是初始 CLIP 偏差在老师侧恢复不了（GateRich Ar2Cl 不加共识的 rich.npz）
- PoE 边际 / 两模型平均边际 只在 0.85–1.11 之间：静态放大很小。
- 第 1 次评估：老师预测 Computer 10 张（真实 99）、Monitor 446 张；学生 12 / 358。不加共识时第 20 次评估恢复到 106 / 90（学生侧 1/频率均衡把它拉回，老师学学生跟着恢复）。
- 加共识：老师学「自己 × 学生」，部分在学自己；老师侧没有类均衡 → 初始偏差锁死 → Computer 107 → 0。
- 修复：--ct_bal（共识的记忆库类边际矫正到学生侧目标分布，无新超参数）→ scripts/run_ctbal.sh。
- 门控信号（tools/diag_rich.py）：Cl2Ar 上 learn / knn / flip_kl 前期也判「学生更准」（真实老师领先），判据不通过；GateRich Re2Ar 有意跳过。

## ct_bal 结果（run_ctbal.sh，后 5）：塌缩修好，但判据不通过
| 任务 | 无共识 | 共识 | ct_bal |
|---|---|---|---|
| Ar2Cl | 78.52 | 75.88 | 77.26 |
| Re2Ar | 84.56 | 85.23 | 83.79 |
| Pr2Cl | 78.10 | 76.20 | 77.15 |
| VisDA 30%（AA） | 88.06 | 88.56 | 88.38 |
被吞类恢复：Ar2Cl Computer 0→58、Marker 6→56；Pr2Cl Clipboards 0→78。Re2Ar 上 Computer 75→48（共识的「吞对方向」也被去掉）。

## 原因分析 3：共识的收益集中在早期（OA 曲线，与无共识之差）
- 第 2–4 次评估：所有任务 +0.7 ~ +2.6（ct_bal 在 Ar2Cl +4.0 / Pr2Cl +2.7）。
- 后期：→Cl / →Pr 转负（Ar2Cl −2.7、Pr2Cl −2.0、Ar2Pr −0.8；ct_bal Ar2Cl −1.3），→Ar 保持正（+0.2 ~ +1.2）。
- ct 塌缩极早：老师 Computer 预测数 34 → 2 → 0（第 0 / 2 / 4 次评估）；无共识时第 10 次评估恢复到 69。
- 与 10-17 交叉点（进度 7.5% ~ 12.5%）一致。→ --ct_until 0.25（沿用 self 冻结点），scripts/run_ctu.sh。GateRich/cap02_tstrong_ct/Ar2Cl 有意中止（ABORTED.txt）。

## ct_bal + ct_until 0.25（run_ctu.sh）：不通过
Ar2Cl 77.47、Pr2Cl 77.79、Re2Ar 84.21、Cl2Ar 83.76（对照 78.52 / 78.10 / 84.56 / 83.77）；VisDA 30% 88.11（88.06）。
早期 +4.0 没保留到最后；ct_bal 的均匀矫正在 Art 去掉收益（Re2Ar Computer 75→45），在 Clipart 矫正过头（Ar2Cl Monitor 69→54）。
共识方向累计已试：teacher / student / both / tcand / rank / topk1-2 / ct_bal / ct_bal+until（DN/VisDA 有效，OH 都不行）。
下一项：只留 ct_until 0.25，不加 ct_bal（run_ctu_nb.sh，18:47 开始）。

## ctu_nb（--consensus teacher --ct_until 0.25，无 ct_bal），seed 42，后 5 均值
- Ar2Cl 77.07（判据 78.52，未过）/ Pr2Cl 78.26（过）/ Re2Ar 85.05（过）/ Cl2Ar 84.86（过）/ VisDA 30% 88.44（过，88.06）
- 4 任务和 +0.29。被吞的类只部分恢复：Ar2Cl Computer 60 / 0 / 15、Pencil 55 / 2 / 25；Pr2Cl Clipboards 78 / 0 / 68（无共识 / 共识 / ctu_nb）
- 解释：塌缩在第 2–4 次评估（进度约 5–10%）就完成了，0.25 切换时学生侧也已经丢掉这些类，1/频率均衡只能慢慢拉回；收益和塌缩同时发生，单靠切换时间分不开
- 筛选只差 Ar2Cl，按最终规则直接跑 12 任务（run_ctu_nb12.sh，21:09 开始），看平均是否 > 86.50；参数不再改
- 12 任务结果（Pr2Re 01:52 中止后按用户要求重跑，中止的目录改名为 Pr2Re_s42_aborted）：cap02_tstrong 86.50 vs ctu 86.51（+0.01，单种子，属于噪声）。→Ar +0.69、→Re +0.10、→Cl −0.40、→Pr −0.36。不通过；共识在 OH 上收尾，OH 不加共识（按数据集配置）。

## ct_marg（--consensus teacher --ct_marg，全程共识 + 预测张数相差 > 3 倍的类不融合），seed 42，后 5 均值
- tools/diag_marg.py 离线：分歧类在 →Cl 上正是被吞的类（Ar2Cl Computer 第 1 次评估就标出），→Ar 几乎不触发；按字面判据不过（Clipboards 第 4 次才标出、Re2Ar −0.01），用户同意直接跑
- Ar2Cl 76.98 ✗ / Pr2Cl 77.42 ✗ / Re2Ar 85.05 ✓ / Cl2Ar 85.08 ✓ / VisDA 30% 88.54 ✓ → 不通过，不跑剩余任务
- 被吞类部分救回：Ar2Cl Computer 0 → 39、Pencil 2 → 31；Pr2Cl Clipboards 0 → 28（无共识 78）。Marker 没救（张数本来就少，比例判不出）
- Ar2Cl 损失分散：最差 8 类 −2.78（Marker、Pencil、Computer、Soda、Shelf、Drill、Refrigerator、Radio），其中 Soda / Shelf / Drill 张数不分歧，是 CLIP 系统性误判
- 推理时师生融合（GateRich rich.npz，后 5）：Ar2Cl 平均 78.58 vs 学生 78.52，Cl2Ar 83.89 vs 83.77，Pr2Cl 78.10 vs 78.09 —— 后期师生收敛，无收益
- 结论：OH 上共识收尾（10 种变体）。解释：COSMO 消融中共识 vs 互相监督 +1.0 解决的是错误吸收 / 锁死，我们用 cap02 + tstrong 已解决（互相监督 86.50），剩下的只有 CLIP 系统偏差的副作用
- 下一步：run_seeds.sh 补 OH 主配置 seed 0 / 1（05:31 开始，约 11 小时）

## 2026-10-29 停补种子（用户：「现在不要补种子没有任何意义需要的是提升acc」），转提准确率
- 离线：推理时时间集成（学生近 5 / 10 / 20 次评估平均）+0.1 ~ +0.2，最高一次评估也只比后 5 高 0.3 ~ 0.4 —— 后期轨迹已平
- 离线：学生 / CLIP / 拼接特征 K=200 聚类 + 源 / CLIP0 / PoE0 / 当前预测命名，全部低于逐样本预测（簇不纯）
- ctu_nb 的 Ar2Cl 损失：Computer / Pencil / Marker −2.3，Shelf / Soda / Drill −0.6，弥散
- run_pc.sh（dba31f6，05:59）：只学提示的老师（--t_prompt --no_adapter）+ 老师侧共识，OH 4 任务 + VisDA 30% + 无共识对照 Ar2Cl / Cl2Ar
  依据：adapter 老师学共识 ≈ 拟合自己（容量大，偏差锁死）；只学提示时老师不塌缩、「只有老师对」是 adapter 的 3 倍（2026_10_21_prompt）
- run_pc.sh 结果（后 5）：只学提示 + 共识 Ar2Cl 76.35 / Re2Ar 80.87 / Cl2Ar 80.48 / Pr2Cl 75.97 / VisDA 30% 87.94；只学提示无共识 Ar2Cl 77.15 / Cl2Ar 80.60 → 不通过
  没有整类被吞（Pr2Cl Clipboards 80，Ar2Cl Marker 37），容量假设对一半；去掉 adapter 老师太弱（→Ar −3 ~ −4，与是否共识无关）
- run_ctp.sh（--t_prompt --ct_prompt）：adapter 学学生 top-3，提示上下文学共识；VisDA 判据提高到 ≥ 88.56（现配置）

## ctp（adapter 学学生 top-3 + 提示上下文学共识，--t_prompt --ct_prompt）结果（10-29）
- Ar2Cl 78.40（判据 78.52，−0.12，不过）；Re2Ar 84.86（判据 84.56，+0.30）。
- Ar2Cl 一挂即判不过，剩余 VisDA / Cl2Ar / Pr2Cl 中止（VisDA 刚开始），未跑。
- 模式与 ctu_nb 相同但幅度更小（→Ar 有收益、→Cl 有损失）：共识换个通道注入，取舍没变，只是被稀释。

## 10-29 错误构成诊断（tools/diag_err.py / diag_prior.py，GateRich/cap02_tstrong，标签只评估）
- 最终错误约 80% 是师生给出同一错标签（Ar2Cl 753/938，Cl2Ar 350/393，Pr2Cl 807/959）；其中约 73% 训练前两边都错。
- 错误分散：前 12 个混淆对只占 22%（Desk_Lamp→Lamp_Shade、TV→Monitor、Telephone→Calculator、Marker→Eraser…）。
- 后处理都无空间：kNN 投票（学生/老师/拼接特征，k=5–50）−1.9~+0.07；Sinkhorn 重分配到无标签边际（unif/tch/mix）±0.1，真实边际上界也只 +1.5；翻转集成 ±0.1。
- →Pr 三个任务终点每类准确率几乎相同（Clipboards 0.7538 ×3、Chair 0.8283 ×3）：终点由 CLIP + 目标数据决定，与源域无关。tstrong 在 →Pr 三个任务都 −0.3~−0.4。
- 结论：师生收敛到同一不动点，上限是表示本身；共识 / 后处理 / 边际都不再追。

## 10-29 学生 EMA（--s_ema 0.5，run_sema.sh，aa55faa）
- 依据：VisDA 主配置 91.60 中 iter 24220 一次瞬时塌陷（train→knife 1595 张，AA 89.00），BN 重估救回到 92.35（archive/2026_10_23_cross），但 OH 上 BN 重估 −0.22。
- EMA（权重 + BN buffer，时间常数 0.5 epoch，事先定）只评估、不改训练；原指标第一次评估逐位复现 52.03。
- 判据：OH 4 任务 EMA 平均 ≥ 原、每任务 ≥ 原 − 0.2；VisDA EMA 后 5 AA ≥ 92.0。OH 通过才跑剩余 8 个任务。
- OH 4 任务结果（原学生逐位复现主配置）：EMA − 原 Ar2Cl +0.12 / Re2Ar +0.09 / Cl2Ar +0.02 / Pr2Cl +0.01，平均 81.24 → 81.30，判据过（幅度很小，基本是后期抖动被平滑）。
- VisDA 中途（11/15 次评估）：原学生逐位复现；稳定期 EMA 比原 AA 高 +0.1~+0.3（12110: 92.02 vs 91.70，13840: 92.19 vs 91.88）。关键点 iter 24220 未到。
- 已排队 run_sema12.sh（OH 剩余 8 任务），VisDA 结束后自动开始（runs/queue_sema12.sh）。
- VisDA 全量结果：原学生后 5 AA 91.60（逐位复现，24220 = 89.00）；EMA 后 5 AA **92.33**（逐次 92.32/92.33/92.32/92.34/92.34，24220 = 92.34），最后一次 92.34。判据 ≥ 92.0 过。与 BN 重估（92.35）几乎相同，但 EMA 在 OH 上不伤（+0.06）。
- 事故：bash -c 的等待进程实际上没死，与 queue_sema12.sh 同时启动了两份 run_sema12.sh（Ar2Pr / Ar2Re 并行，显存 32.1G）；14:55 杀掉后者，Ar2Re 半成品改名 Ar2Re_s42_dupkilled，保留的那份之后会正常跑 Ar2Re。

## 10-29 OH 12 任务 + 学生 EMA（run_sema12.sh 汇总）
原学生 12/12 与主配置逐位相同，均值 86.50；EMA 均值 86.53（COSMO 86.41）。10/12 不降，Cl2Re −0.08、Cl2Pr −0.02，属噪声范围。
判定没问题 → 22:12 启动 DN-126 12 任务：VARIANT=self_ct_sema（runs/dn_sema.log，组 cap02_tstrong_self_ct_sema_e10）。

## 10-30 DN-126 12 任务 + 学生 EMA（seed 42，最终版）
原学生 12/12 逐位复现 self_ct（86.00）；EMA 86.09（COSMO 85.1），12/12 全部提升 +0.05~+0.13。→real 3 个任务仍低 COSMO 0.9~1.2。
最终 seed 42：OH 86.53 / VisDA 92.33 / DN 86.09。
之后进入多种子（run_seeds_final.sh，种子 0 / 1，每个种子重训源模型）：gpuA VisDA s0 → s1 → DN s0；gpuB OH s0 → s1 → DN s1。

## 10-31 排期（用户 10-31 全天不在；3 个种子已定为 42 / 0 / 1）
- 多种子：gpuB OH s0 → OH s1（约 10-31 12:00 CST 完）；gpuA DN s0 → DN s1（约 10-31 18:00 CST 完）。
- 种子跑完后按 scripts/run_ablation_final.sh（commit 见 git log）跑累积消融，用户说不用马上开、跑完种子再做：
  - gpuA：ABL_SET=oh（R0 VSFOT、R1 +kNN，各 12 任务，约 21 小时）。R2–R5 已有组都在 gpuA，torch 2.8，必须同机。
  - gpuB：ABL_SET=visda（自训 seed 42 源模型 Source_visda_b42，r7 → r0…r6，约 17 小时）。gpuB 是 torch 2.10，与 gpuA 不能逐位比，所以整条链在本机自洽。
  - DN 不新跑，用已有组。
- 两台 torch 不同（gpuA 2.8.0+cu128，gpuB 2.10.0+cu129）：OH s0/s1 在 gpuB，其余在 gpuA，写论文时注明。

## 2026-10-31 gpuB（31435）被抢占：OH s0 / s1 结果只剩汇总数字（本地留底）
gpuB：torch 2.10.0+cu129，commit 0a0f7d9，源模型 runs/Source_home_s<seed>/ 本机按种子训练；输出目录 / final.pt 已无法访问。
EMA 学生 OA 后 5 次均值（括号内为原学生），摘自 gpuB 上 run_seeds_final.sh 的汇总：

| 任务 | s0 | s1 |
|---|---|---|
| Ar2Cl | 78.04（78.05） | 78.32（78.27） |
| Ar2Pr | 91.99（92.00） | 91.68（91.77） |
| Ar2Re | 91.77（91.67） | 91.59（91.52） |
| Cl2Ar | 84.26（84.14） | 83.35（83.45） |
| Cl2Pr | 92.10（92.03） | 91.63（91.59） |
| Cl2Re | 91.53（91.51） | 91.37（91.39） |
| Pr2Ar | 85.08（85.03） | 84.37（84.50） |
| Pr2Cl | 78.46（78.36） | 78.06（78.07） |
| Pr2Re | 91.68（91.65） | 未抄到；由平均反推约 91.72（汇总平均四舍五入，误差 ±0.06） |
| Re2Ar | 84.82（84.78） | 84.49（84.38） |
| Re2Cl | 77.66（77.57） | 75.99（75.85） |
| Re2Pr | 92.88（92.89） | 91.95（91.99） |
| 平均 | 86.69 | 86.21 |

OH 3 种子（EMA）：s42 86.53（gpuA）/ s0 86.69 / s1 86.21 → 86.48 ± 0.24（COSMO 86.41）。
另：gpuB 上 seed 42 Art 源模型复查 →Cl/Pr/Re 45.1 / 69.5 / 77.5，gpuA 为 44.6 / 69.2 / 77.3（跨 torch 版本不逐位一致，差 0.2–0.5）。
