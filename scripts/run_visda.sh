#!/usr/bin/env bash
# =============================================================================
# VisDA-C 快速检查（2026-10-23），synthesis → real，ResNet-101，seed 42
#   Office-Home 上 cap02 + tstrong 12 任务平均 86.50（COSMO 86.41），收益集中在 CLIP 零样本最弱的 Clipart 目标。
#   VisDA-C 的 real 域对 CLIP 比较熟悉，这里先看现有方法在 VisDA-C 上和 COSMO（91.4）/ VSFOT 公开数字（90.6）差多少。
#   源模型：bash 里先跑
#     python trainSource.py --name Source_visda/Source_synthesis --dataset VisDA --source synthesis --arch resnet101 --lr 1e-3 --epochs 2
#   EPOCHS 默认 3、BATCH 默认 32（快速设置；不是论文设置）：ResNet-101 在 batch 64 下单卡 32G 显存放不下（2026-10-23 两次 OOM）。
#   VisDA-C 的指标看各类平均准确率（AA），OA 一并列出。
#
#   !bash scripts/run_visda.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_visda.sh
# =============================================================================

set -euo pipefail

DATASET="VisDA"
PYTHON="python -u"
GPU=0
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-32}"

BASE="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12"
CAP="--align_weight cap --align_cap 0.2"

# 训练任务，格式 "组名|额外参数"；按重要性排序
GRPS=(
    "cap02_tstrong_e${EPOCHS}_b${BATCH}|--t_strong"
    "cap02_e${EPOCHS}_b${BATCH}|"
)
TRAIN_ROOT="VisDA"
TASK="S2R"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

if [ ! -f "runs/Source_visda/Source_synthesis/Source.pt" ]; then
    echo "[错误] 没有源模型 runs/Source_visda/Source_synthesis/Source.pt"; exit 1
fi

for g in "${GRPS[@]}"; do
    name="${g%%|*}"; extra="${g#*|}"
    dir="runs/$TRAIN_ROOT/$name/${TASK}_s42"
    [ "$SUMMARY_ONLY" = "1" ] && continue
    if [ -f "$dir/.done" ]; then echo "==> $name：已完成，跳过"; continue; fi
    if [ -d "$dir" ]; then echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1; fi
    echo "==> $name  参数：$BASE $CAP $extra"
    $PYTHON main.py --device "$GPU" --name "$TRAIN_ROOT/$name/${TASK}_s42" --epochs "$EPOCHS" --batch_size "$BATCH" --dataset "$DATASET" \
        --source synthesis --target real --seed 42 $BASE $CAP $extra
    touch "$dir/.done"
done

echo "============================== 汇总开始（把这一段完整贴回）=============================="
echo "## VisDA-C synthesis → real（ResNet-101，${EPOCHS} epoch，batch ${BATCH}，seed 42，最后 5 次评估均值）"
echo ""
echo "| 组 | AA（各类平均） | OA | 状态 |"
echo "|---|---|---|---|"
for g in "${GRPS[@]}"; do
    name="${g%%|*}"; dir="runs/$TRAIN_ROOT/$name/${TASK}_s42"
    [ -f "$dir/Detail.txt" ] || { echo "| $name | - | - | 未开始 |"; continue; }
    aa=$(grep -o "Average Accuracy (AA): [0-9.]*" "$dir/Detail.txt" | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100}')
    oa=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$dir/Detail.txt" | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100}')
    st=$([ -f "$dir/.done" ] && echo 完成 || echo 进行中)
    echo "| $name | $aa | $oa | $st |"
done
echo ""
echo "参考：COSMO 91.4，VSFOT 公开数字 90.6（论文表格，AA）"
echo "各类准确率（最后一次评估）："
for g in "${GRPS[@]}"; do
    name="${g%%|*}"; f="runs/$TRAIN_ROOT/$name/${TASK}_s42/ModellineAcc.txt"
    [ -f "$f" ] && { echo "$name"; head -1 "$f"; tail -1 "$f"; }
done
echo "============================== 汇总结束 =============================="
