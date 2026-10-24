#!/usr/bin/env bash
# =============================================================================
# DomainNet-126 检查（2026-10-23），ResNet-50，seed 42
#   担心：Office-Home 上的收益集中在 CLIP 最弱的 Clipart 目标；类别最不均衡的 Art 目标输 COSMO 最多（小类被多预测 15 ~ 19%），
#   DomainNet 的 clipart / painting / sketch 比 Art 更不均衡（CV 0.60 ~ 0.63，Art 0.52）。这里看现有方法在 DomainNet-126 上的实际差距。
#   源模型：python trainSource.py --name Source_domainnet/Source_<域> --dataset DomainNet --source <域> --arch resnet50 --epochs 5
#   EPOCHS 默认 10（快速设置，不是论文设置）。COSMO 论文表格是 12 个任务（OA），按优先级排：先跑不均衡的目标（sketch / clipart / painting），
#   real 目标（6.9 万张，最慢）放最后。
#
#   !bash scripts/run_domainnet.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_domainnet.sh
#   只跑前 n 个任务：!N_TASKS=3 bash scripts/run_domainnet.sh
#   VARIANT=self_ct：self + --consensus teacher（老师学老师 × 学生的熵加权共识；30% 子集 R2P / C2S / S2P +1.6 ~ 2.2），对照列为 self
#   VARIANT=self：加 --clsw_mode self（进度 0.25 冻结学生平均概率作均衡目标；R→S 77.95 → 79.43），汇总里列出同任务的原方法作对照
# =============================================================================

set -euo pipefail

DATASET="DomainNet"
PYTHON="python -u"
GPU=0
EPOCHS="${EPOCHS:-10}"
N_TASKS="${N_TASKS:-12}"

BASE="--align_mode ot --knn_k 10 --align_view both --arch resnet50 --workers 12"
CAP="--align_weight cap --align_cap 0.2"
VARIANT="${VARIANT:-}"
BASE_GROUP="cap02_tstrong_e${EPOCHS}"
GROUP="cap02_tstrong${VARIANT:+_$VARIANT}_e${EPOCHS}"
EXTRA="--t_strong"
case "$VARIANT" in
    "") ;;
    self) EXTRA="$EXTRA --clsw_mode self" ;;
    self_ct) EXTRA="$EXTRA --clsw_mode self --consensus teacher"; BASE_GROUP="cap02_tstrong_self_e${EPOCHS}" ;;
    *) echo "未知 VARIANT: $VARIANT"; exit 1 ;;
esac

# 源域:目标域:COSMO:VSFOT（论文表格，https://arxiv.org/html/2608.04604，CLIP ViT-B/32，OA）
PAIRS=(real:sketch:79.8:76.9 real:clipart:85.4:83.3 painting:sketch:80.2:77.5 clipart:sketch:79.7:77.2
       sketch:clipart:84.7:83.5 painting:clipart:84.8:83.0 real:painting:82.8:79.8 clipart:painting:82.8:79.5
       sketch:painting:82.9:79.6 painting:real:92.7:90.5 clipart:real:92.7:90.4 sketch:real:92.7:90.4)
TRAIN_ROOT="DomainNet"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

abbr() { echo "$1" | cut -c1 | tr a-z A-Z; }

k=0
for pair in "${PAIRS[@]}"; do
    k=$((k + 1)); [ "$k" -gt "$N_TASKS" ] && break
    IFS=: read -r s t _ _ <<< "$pair"; task="$(abbr "$s")2$(abbr "$t")"
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
echo "## DomainNet-126 ${GROUP}（ResNet-50，${EPOCHS} epoch，seed 42；均值 = 不含第 0 步的最后至多 5 次评估）"
echo ""
echo "| 任务 | 源模型（第 0 步） | OA 均值 | OA 最后一次 | AA 最后一次 | 对照 $BASE_GROUP | COSMO | VSFOT（公开） | − COSMO | 状态 |"
echo "|---|---|---|---|---|---|---|---|---|---|"
for pair in "${PAIRS[@]}"; do
    IFS=: read -r s t cosmo vsfot <<< "$pair"; task="$(abbr "$s")2$(abbr "$t")"
    f="runs/$TRAIN_ROOT/$GROUP/${task}_s42/Detail.txt"
    fb="runs/$TRAIN_ROOT/$BASE_GROUP/${task}_s42/Detail.txt"
    ob=$([ -f "$fb" ] && grep -o "Overall Accuracy (OA): [0-9.]*" "$fb" | tail -n +2 | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100}' || echo -)
    [ -f "$f" ] || { echo "| $task | - | - | - | - | $ob | $cosmo | $vsfot | - | 未开始 |"; continue; }
    oa0=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$f" | head -1 | awk '{printf "%.2f", $NF*100}')
    oa=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$f" | tail -n +2 | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100}')
    oal=$(grep -o "Overall Accuracy (OA): [0-9.]*" "$f" | tail -n +2 | tail -1 | awk '{printf "%.2f", $NF*100}')
    aal=$(grep -o "Average Accuracy (AA): [0-9.]*" "$f" | tail -n +2 | tail -1 | awk '{printf "%.2f", $NF*100}')
    st=$([ -f "${f%/*}/.done" ] && echo 完成 || echo 进行中)
    dc=$(awk -v a="$oa" -v b="$cosmo" 'BEGIN {if (a != "") printf "%+.2f", a - b; else printf "-"}')
    echo "| $task | $oa0 | $oa | $oal | $aal | $ob | $cosmo | $vsfot | $dc | $st |"
done
echo ""
echo "参考：COSMO DomainNet-126 12 任务平均 85.1，VSFOT 公开数字 82.6（论文表格）"
echo "============================== 汇总结束 =============================="
