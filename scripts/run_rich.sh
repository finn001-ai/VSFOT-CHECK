#!/usr/bin/env bash
# =============================================================================
# 2026-10-28 门控信息收集：cap02_tstrong（不加共识）+ --log_dynamics --log_samples --log_rich，40 epoch，seed 42，串行
#   Dawid–Skene 门控（tools/diag_ds.py）只用硬预测 / 熵，Ar2Cl 上全部判反；这里记完整概率（含水平翻转）和两边特征，离线找能区分
#   「学生领先」（Ar2Cl、Pr2Cl：加共识 −2.64 / −1.90）和「老师领先」（Cl2Ar、Re2Ar：加共识 +1.18 / +0.67）的量。
#   训练过程与 cap02_tstrong 相同（记录不影响随机数和参数），最后 5 次 OA 应与 78.52 / 78.10 / 83.77 / 84.56 一致，作为校验。
#
#   !bash scripts/run_rich.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
GROUP="GateRich/cap02_tstrong"
EXTRA="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --log_dynamics --log_samples --log_rich"

PAIRS="art:clipart clipart:art product:clipart realworld:art"
abbr() { case "$1" in art) echo Ar ;; clipart) echo Cl ;; product) echo Pr ;; realworld) echo Re ;; esac; }
mkdir -p runs/rich_logs

T0=$(date +%s)
for pair in $PAIRS; do
    s="${pair%%:*}"; t="${pair##*:}"; task="$(abbr "$s")2$(abbr "$t")"
    dir="$GROUP/${task}_s42"
    if [ -f "runs/$dir/.done" ]; then echo "==> $task：已完成，跳过"; continue; fi
    if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
    echo "==> 开始 $task（$(date +%H:%M)）"
    # $EXTRA 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个任务
    if $PYTHON main.py --device "$GPU" --name "$dir" --dataset OfficeHomeDataset --source "$s" --target "$t" --seed 42 $EXTRA \
        > "runs/rich_logs/${task}.log" 2>&1; then
        touch "runs/$dir/.done"; echo "==> 完成 $task"
    else
        echo "[错误] $task 训练失败，见 runs/rich_logs/${task}.log"
    fi
done
echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"

echo "============================== 汇总开始 =============================="
$PYTHON - <<'PY'
import os, re
ref = {"Ar2Cl": 78.52, "Cl2Ar": 83.77, "Pr2Cl": 78.10, "Re2Ar": 84.56}
for t, r in ref.items():
    p = f"runs/GateRich/cap02_tstrong/{t}_s42/Detail.txt"
    if not os.path.exists(p):
        print(t, "-"); continue
    v = [float(x) * 100 for x in re.findall(r"Overall Accuracy \(OA\): ([0-9.]+)", open(p).read())][1:]
    print(f"{t}: 后 5 {sum(v[-5:]) / len(v[-5:]):.2f}（cap02_tstrong {r}），rich.npz {os.path.exists(os.path.dirname(p) + '/rich.npz')}")
PY
echo "============================== 汇总结束 =============================="
