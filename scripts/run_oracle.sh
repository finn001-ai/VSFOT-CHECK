#!/usr/bin/env bash
# =============================================================================
# 类先验上界（2026-10-23）：--clsw_mode oracle，类均衡的目标分布换成目标域真实类频率（用了标签，诊断，不是方法）
#   1/频率重加权把学生的预测分布推向均匀：Office-Home Art 目标上小类被多预测 15 ~ 19%（输 COSMO 最多的目标域），
#   VisDA-C 上最大的类 car（10401 张，占 18.8%）只被预测出 0.67 倍，1960 张判成 truck，car 准确率 62%。
#   之前的无标签先验（CLIP 零样本平均概率，archive/2026_10_20_ab）−1.33，因为 CLIP 的先验本身偏 8 ~ 19 倍。
#   这里先测「先验完全正确」能拿多少：上界小就不走这条路，上界大再找无标签的估计方法。
#     Office-Home：Ar→Cl、Ar→Pr、Cl→Ar，cap02 + tstrong，40 epoch，对照 run_tstrong12.sh 的 cap02_tstrong（78.52 / 91.63 / 83.77）
#     VisDA-C：cap02 + tstrong，3 epoch、batch 32，对照 run_visda.sh 的 cap02_tstrong_e3_b32（AA 88.64）
#
#   !bash scripts/run_oracle.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_oracle.sh
# =============================================================================

set -euo pipefail

PYTHON="python -u"
GPU=0
BASE="--align_mode ot --knn_k 10 --align_view both --workers 12"
CAP="--align_weight cap --align_cap 0.2"
ORACLE="--t_strong --clsw_mode oracle"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

# 训练任务，格式 "目录|数据集|源域|目标域|额外参数"
JOBS=(
    "WeightSched/cap02_tstrong_oracle/Cl2Ar_s42|OfficeHomeDataset|clipart|art|--epochs 40"
    "VisDA/cap02_tstrong_oracle_e3_b32/S2R_s42|VisDA|synthesis|real|--epochs 3 --batch_size 32 --arch resnet101"
    "WeightSched/cap02_tstrong_oracle/Ar2Cl_s42|OfficeHomeDataset|art|clipart|--epochs 40"
    "WeightSched/cap02_tstrong_oracle/Ar2Pr_s42|OfficeHomeDataset|art|product|--epochs 40"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r dir ds s t extra <<< "$job"
    [ "$SUMMARY_ONLY" = "1" ] && continue
    if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
    if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1; fi
    echo "==> $dir  参数：$BASE $CAP $ORACLE $extra"
    $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 $BASE $CAP $ORACLE $extra
    touch "runs/$dir/.done"
done

# 每格 = 不含第 0 步的最后至多 5 次评估均值（OA）；VisDA 另列 AA
last5() { grep -o "$2: [0-9.]*" "$1" | tail -n +2 | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100; else printf "-"}'; }
echo "============================== 汇总开始（把这一段完整贴回）=============================="
echo "## 类先验上界（seed 42；oracle = 真实类频率作为均衡目标，用了标签）"
echo ""
echo "| 任务 | 对照 cap02_tstrong | oracle | 差 |"
echo "|---|---|---|---|"
for t in Ar2Cl Ar2Pr Cl2Ar; do
    b="runs/WeightSched/cap02_tstrong/${t}_s42/Detail.txt"; o="runs/WeightSched/cap02_tstrong_oracle/${t}_s42/Detail.txt"
    bv=$([ -f "$b" ] && last5 "$b" "Overall Accuracy (OA)" || echo -); ov=$([ -f "$o" ] && last5 "$o" "Overall Accuracy (OA)" || echo -)
    d=$(awk -v a="$ov" -v b="$bv" 'BEGIN {if (a != "-" && b != "-") printf "%+.2f", a - b; else printf "-"}')
    echo "| $t（OA） | $bv | $ov | $d |"
done
for m in "Average Accuracy (AA)" "Overall Accuracy (OA)"; do
    b="runs/VisDA/cap02_tstrong_e3_b32/S2R_s42/Detail.txt"; o="runs/VisDA/cap02_tstrong_oracle_e3_b32/S2R_s42/Detail.txt"
    bv=$([ -f "$b" ] && last5 "$b" "$m" || echo -); ov=$([ -f "$o" ] && last5 "$o" "$m" || echo -)
    d=$(awk -v a="$ov" -v b="$bv" 'BEGIN {if (a != "-" && b != "-") printf "%+.2f", a - b; else printf "-"}')
    echo "| VisDA S2R（${m%% *}） | $bv | $ov | $d |"
done
echo "============================== 汇总结束 =============================="
