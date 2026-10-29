#!/usr/bin/env bash
# =============================================================================
# 2026-10-29 VisDA-C 主配置 norm65_self_ct_e15_b32 补种子（seed 0、1；seed 42 已有），15 epoch（COSMO 协议），batch 32，串行
#   seed 42：AA 后 5 均值 91.60，COSMO 91.4（3 个种子）。单种子 +0.20，补种子才能下结论。参数与 seed 42 的 argsInfo 相同，只换种子。
#   每个约 2.6 小时；排在 run_seeds.sh（OH）之后。
#
#   !bash scripts/run_vseeds.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_vseeds.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
GROUP="VisDA/norm65_self_ct_e15_b32"
EXTRA="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 15 --batch_size 32 --clsw_mode self --align_norm_c 65 --consensus teacher"
mkdir -p runs/vseeds_logs

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    for seed in 0 1; do
        dir="$GROUP/S2R_s${seed}"
        if [ -f "runs/$dir/.done" ]; then echo "==> S2R s$seed：已完成，跳过"; continue; fi
        if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
        echo "==> 开始 S2R s$seed（$(date +%H:%M)）"
        if $PYTHON main.py --device "$GPU" --name "$dir" --dataset VisDA --source synthesis --target real --seed "$seed" $EXTRA \
            > "runs/vseeds_logs/S2R_s${seed}.log" 2>&1; then
            touch "runs/$dir/.done"; echo "==> 完成 S2R s$seed"
        else
            echo "[错误] S2R s$seed 训练失败，见 runs/vseeds_logs/S2R_s${seed}.log"
        fi
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'PY'
import os, re
import numpy as np
print("## VisDA-C norm65_self_ct_e15_b32 多种子（AA / OA 后 5 = 最后 5 次评估均值；COSMO AA 91.4，3 个种子）\n")
aa = []
for s in (42, 0, 1):
    d = f"runs/VisDA/norm65_self_ct_e15_b32/S2R_s{s}"
    if not os.path.exists(f"{d}/Detail.txt") or (s != 42 and not os.path.exists(f"{d}/.done")):
        print(f"seed {s}：未完成"); continue
    t = open(f"{d}/Detail.txt").read()
    A = [float(x) * 100 for x in re.findall(r"Average Accuracy \(AA\): ([0-9.]+)", t)][1:]
    O = [float(x) * 100 for x in re.findall(r"Overall Accuracy \(OA\): ([0-9.]+)", t)][1:]
    aa.append(np.mean(A[-5:]))
    print(f"seed {s}：AA 后 5 {aa[-1]:.2f}，最后 {A[-1]:.2f}，最高 {max(A):.2f}；OA 后 5 {np.mean(O[-5:]):.2f}")
    p = f"{d}/ModellineAcc.txt"
    if os.path.exists(p):
        L = open(p).read().strip().splitlines(); print("  ", L[0]); print("  ", L[-1])
if len(aa) > 1:
    print(f"\n{len(aa)} 个种子 AA：{np.mean(aa):.2f} ± {np.std(aa, ddof=1):.2f}（COSMO 91.4）")
PY
echo "============================== 汇总结束 =============================="
