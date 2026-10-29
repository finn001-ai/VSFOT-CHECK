#!/usr/bin/env bash
# =============================================================================
# 2026-10-29 Office-Home 主配置 cap02_tstrong 补种子（seed 0、1；seed 42 已有），40 epoch，串行
#   seed 42：12 任务平均 86.50，COSMO 86.41 ± 0.084（3 个种子，逐任务标准差平均 0.199）。单种子 +0.09 在噪声范围内，补种子才能下结论。
#   老师侧共识在 OH 上的 10 种变体都不超过不加共识（archive/2026_10_28_q1028），OH 主配置定为不加共识。参数与 run_tstrong12.sh 相同，只换种子。
#   顺序：先跑完 seed 0 的 12 个任务（约 5.5 小时，可先看两种子平均），再跑 seed 1。
#
#   !bash scripts/run_seeds.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_seeds.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
GROUP="cap02_tstrong"
EXTRA="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong"

PAIRS="art:clipart art:product art:realworld clipart:art clipart:product clipart:realworld \
product:art product:clipart product:realworld realworld:art realworld:clipart realworld:product"
abbr() { case "$1" in art) echo Ar ;; clipart) echo Cl ;; product) echo Pr ;; realworld) echo Re ;; esac; }
mkdir -p runs/seeds_logs

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    for seed in 0 1; do
        for pair in $PAIRS; do
            s="${pair%%:*}"; t="${pair##*:}"; task="$(abbr "$s")2$(abbr "$t")"
            dir="WeightSched/$GROUP/${task}_s${seed}"
            if [ -f "runs/$dir/.done" ]; then echo "==> $task s$seed：已完成，跳过"; continue; fi
            if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
            echo "==> 开始 $task s$seed（$(date +%H:%M)）"
            # $EXTRA 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个任务
            if $PYTHON main.py --device "$GPU" --name "$dir" --dataset OfficeHomeDataset --source "$s" --target "$t" --seed "$seed" $EXTRA \
                > "runs/seeds_logs/${task}_s${seed}.log" 2>&1; then
                touch "runs/$dir/.done"; echo "==> 完成 $task s$seed"
            else
                echo "[错误] $task s$seed 训练失败，见 runs/seeds_logs/${task}_s${seed}.log"
            fi
        done
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'PY'
import os, re
import numpy as np
cosmo = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Ar2Re": 92.0, "Cl2Ar": 85.7, "Cl2Pr": 92.4, "Cl2Re": 91.9,
         "Pr2Ar": 84.3, "Pr2Cl": 76.1, "Pr2Re": 91.7, "Re2Ar": 85.6, "Re2Cl": 75.5, "Re2Pr": 92.7}
def last5(t, seed):
    d = f"runs/WeightSched/cap02_tstrong/{t}_s{seed}"
    if not os.path.exists(f"{d}/.done") and seed != 42:
        return None
    v = [float(x) * 100 for x in re.findall(r"Overall Accuracy \(OA\): ([0-9.]+)", open(f"{d}/Detail.txt").read())][1:] if os.path.exists(f"{d}/Detail.txt") else []
    return sum(v[-5:]) / len(v[-5:]) if v else None
seeds = (42, 0, 1)
R = {t: [last5(t, s) for s in seeds] for t in cosmo}
f = lambda v: "-" if v is None else f"{v:.2f}"
print("## Office-Home cap02_tstrong 多种子（40 epoch，OA 后 5 = 最后 5 次评估均值）\n")
print("| 任务 | s42 | s0 | s1 | 均值 ± 标准差 | COSMO | 差 |")
print("|---|---|---|---|---|---|---|")
for t, r in R.items():
    v = [x for x in r if x is not None]
    m = f"{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}" if v else "-"
    print(f"| {t} | {' | '.join(f(x) for x in r)} | {m} | {cosmo[t]} | {np.mean(v) - cosmo[t]:+.2f} |" if v else f"| {t} | - | - | - | - | {cosmo[t]} | - |")
print()
avgs = []
for i, s in enumerate(seeds):
    v = [R[t][i] for t in cosmo]
    if all(x is not None for x in v):
        avgs.append(np.mean(v)); print(f"seed {s}：12 任务平均 {avgs[-1]:.2f}")
    else:
        done = [t for t in cosmo if R[t][i] is not None]
        if done:
            print(f"seed {s}：已完成 {len(done)} / 12，这些任务上 {np.mean([R[t][i] for t in done]):.2f}（seed 42 同任务 {np.mean([R[t][0] for t in done]):.2f}，COSMO {np.mean([cosmo[t] for t in done]):.2f}）")
if len(avgs) > 1:
    print(f"\n{len(avgs)} 个种子：{np.mean(avgs):.2f} ± {np.std(avgs, ddof=1):.3f}（COSMO 86.41 ± 0.084，3 个种子）")
PY
echo "============================== 汇总结束 =============================="
