#!/usr/bin/env bash
# =============================================================================
# DomainNet-126 检查（2026-10-23），ResNet-50，seed 42
#   担心：Office-Home 上的收益集中在 CLIP 最弱的 Clipart 目标；类别最不均衡的 Art 目标输 COSMO 最多（小类被多预测 15 ~ 19%），
#   DomainNet 的 clipart / painting / sketch 比 Art 更不均衡（CV 0.60 ~ 0.63，Art 0.52）。这里看现有方法在 DomainNet-126 上的实际差距。
#   源模型：python trainSource.py --name Source_domainnet/Source_<域> --dataset DomainNet --source <域> --arch resnet50 --epochs 5
#   EPOCHS 默认 10（快速设置，不是论文设置）。7 个标准任务按优先级排：先跑两个最不均衡的目标（R→S、R→C），再 P→R 对照，其余在后。
#
#   !bash scripts/run_domainnet.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_domainnet.sh
#   只跑前 n 个任务：!N_TASKS=3 bash scripts/run_domainnet.sh
# =============================================================================

set -euo pipefail

DATASET="DomainNet"
PYTHON="python -u"
GPU=0
EPOCHS="${EPOCHS:-10}"
N_TASKS="${N_TASKS:-7}"

BASE="--align_mode ot --knn_k 10 --align_view both --arch resnet50 --workers 12"
CAP="--align_weight cap --align_cap 0.2"
GROUP="cap02_tstrong_e${EPOCHS}"
EXTRA="--t_strong"

# 源域:目标域；COSMO（论文表格）各任务数字没有记录在本地，汇总只给平均 85.1 作参考
PAIRS=(real:sketch real:clipart painting:real real:painting painting:clipart clipart:sketch sketch:painting)
TRAIN_ROOT="DomainNet"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

abbr() { echo "$1" | cut -c1 | tr a-z A-Z; }

k=0
for pair in "${PAIRS[@]}"; do
    k=$((k + 1)); [ "$k" -gt "$N_TASKS" ] && break
    s="${pair%%:*}"; t="${pair##*:}"; task="$(abbr "$s")2$(abbr "$t")"
    dir="runs/$TRAIN_ROOT/$GROUP/${task}_s42"
    [ "$SUMMARY_ONLY" = "1" ] && continue
    if [ ! -f "runs/Source_domainnet/Source_$s/Source.pt" ]; then echo "[错误] 没有源模型 Source_$s"; exit 1; fi
    if [ -f "$dir/.done" ]; then echo "==> $task：已完成，跳过"; continue; fi
    if [ -d "$dir" ]; then echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1; fi
    echo "==> $task（$s → $t）  参数：$BASE $CAP $EXTRA"
    $PYTHON main.py --device "$GPU" --name "$TRAIN_ROOT/$GROUP/${task}_s42" --epochs "$EPOCHS" --dataset "$DATASET" \
        --source "$s" --target "$t" --seed 42 $BASE $CAP $EXTRA
    touch "$dir/.done"
done

echo "============================== 汇总开始（把这一段完整贴回）=============================="
echo "## DomainNet-126（ResNet-50，${EPOCHS} epoch，seed 42；均值 = 不含第 0 步的最后至多 5 次评估）"
echo ""
echo "| 任务 | 源模型（第 0 步） | OA 均值 | OA 最后一次 | AA 最后一次 | 状态 |"
echo "|---|---|---|---|---|---|"
for pair in "${PAIRS[@]}"; do
    s="${pair%%:*}"; t="${pair##*:}"; task="$(abbr "$s")2$(abbr "$t")"
    f="runs/$TRAIN_ROOT/$GROUP/${task}_s42/Detail.txt"
    [ -f "$f" ] || { echo "| $task | - | - | - | - | 未开始 |"; continue; }
    oa0=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$f" | head -1 | awk '{printf "%.2f", $NF*100}')
    oa=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$f" | tail -n +2 | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100}')
    oal=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$f" | tail -n +2 | tail -1 | awk '{printf "%.2f", $NF*100}')
    aal=$(grep -o "Average Accuracy (AA): [0-9.]*" "$f" | tail -n +2 | tail -1 | awk '{printf "%.2f", $NF*100}')
    st=$([ -f "${f%/*}/.done" ] && echo 完成 || echo 进行中)
    echo "| $task | $oa0 | $oa | $oal | $aal | $st |"
done
echo ""
echo "参考：COSMO DomainNet-126 平均 85.1，VSFOT 公开数字 82.6（论文表格）"
echo "============================== 汇总结束 =============================="
