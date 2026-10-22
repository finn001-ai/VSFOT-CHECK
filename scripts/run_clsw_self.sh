#!/usr/bin/env bash
# =============================================================================
# 无标签类先验（2026-10-23）：--clsw_mode self，进度 0.25 之前同原方法（1/频率，推向均匀），到 0.25 时把学生记忆库的平均概率冻结为
#   均衡目标，之后推向它。规则事先定好（0.25 来自快照诊断：此时各种无标签估计与真实先验的 L1 已稳定在均匀分布的 1/3 ~ 2/3），不按测试结果调。
#   上界（run_oracle.sh，真实先验）：Cl→Ar +0.96、Ar→Cl +1.37、VisDA AA 最后一次 +0.82。
#     Office-Home：Ar→Cl、Ar→Pr、Cl→Ar，40 epoch，对照 cap02_tstrong（78.52 / 91.63 / 83.77）
#     VisDA-C：3 epoch、batch 32，对照 cap02_tstrong_e3_b32（AA 88.64）；DomainNet R→S：10 epoch，对照 77.95
#
#   !bash scripts/run_clsw_self.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_clsw_self.sh
# =============================================================================

set -euo pipefail

PYTHON="python -u"
GPU=0
BASE="--align_mode ot --knn_k 10 --align_view both --workers 12"
CAP="--align_weight cap --align_cap 0.2"
ORACLE="--t_strong --clsw_mode self"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

# 训练任务，格式 "目录|数据集|源域|目标域|额外参数"
JOBS=(
    "WeightSched/cap02_tstrong_self/Ar2Cl_s42|OfficeHomeDataset|art|clipart|--epochs 40"
    "WeightSched/cap02_tstrong_self/Ar2Pr_s42|OfficeHomeDataset|art|product|--epochs 40"
    "WeightSched/cap02_tstrong_self/Cl2Ar_s42|OfficeHomeDataset|clipart|art|--epochs 40"
    "VisDA/cap02_tstrong_self_e3_b32/S2R_s42|VisDA|synthesis|real|--epochs 3 --batch_size 32 --arch resnet101"
    "DomainNet/cap02_tstrong_self_e10/R2S_s42|DomainNet|real|sketch|--epochs 10 --arch resnet50"
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
echo "## 无标签类先验 self（seed 42；进度 0.25 冻结学生平均概率作为均衡目标）"
echo ""
echo "| 任务 | 对照 cap02_tstrong | self | 差 | 上界 oracle |"
echo "|---|---|---|---|"
for t in Ar2Cl Ar2Pr Cl2Ar; do
    b="runs/WeightSched/cap02_tstrong/${t}_s42/Detail.txt"; o="runs/WeightSched/cap02_tstrong_self/${t}_s42/Detail.txt"
    bv=$([ -f "$b" ] && last5 "$b" "Overall Accuracy (OA)" || echo -); ov=$([ -f "$o" ] && last5 "$o" "Overall Accuracy (OA)" || echo -)
    d=$(awk -v a="$ov" -v b="$bv" 'BEGIN {if (a != "-" && b != "-") printf "%+.2f", a - b; else printf "-"}')
    q="runs/WeightSched/cap02_tstrong_oracle/${t}_s42/Detail.txt"; qv=$([ -f "$q" ] && last5 "$q" "Overall Accuracy (OA)" || echo -)
    echo "| $t（OA） | $bv | $ov | $d | $qv |"
done
for m in "Average Accuracy (AA)" "Overall Accuracy (OA)"; do
    b="runs/VisDA/cap02_tstrong_e3_b32/S2R_s42/Detail.txt"; o="runs/VisDA/cap02_tstrong_self_e3_b32/S2R_s42/Detail.txt"
    bv=$([ -f "$b" ] && last5 "$b" "$m" || echo -); ov=$([ -f "$o" ] && last5 "$o" "$m" || echo -)
    d=$(awk -v a="$ov" -v b="$bv" 'BEGIN {if (a != "-" && b != "-") printf "%+.2f", a - b; else printf "-"}')
    q="runs/VisDA/cap02_tstrong_oracle_e3_b32/S2R_s42/Detail.txt"; qv=$([ -f "$q" ] && last5 "$q" "$m" || echo -)
    echo "| VisDA S2R（${m%% *}） | $bv | $ov | $d | $qv |"
done
b="runs/DomainNet/cap02_tstrong_e10/R2S_s42/Detail.txt"; o="runs/DomainNet/cap02_tstrong_self_e10/R2S_s42/Detail.txt"; q="runs/DomainNet/cap02_tstrong_oracle_e10/R2S_s42/Detail.txt"
bv=$([ -f "$b" ] && last5 "$b" "Overall Accuracy (OA)" || echo -); ov=$([ -f "$o" ] && last5 "$o" "Overall Accuracy (OA)" || echo -); qv=$([ -f "$q" ] && last5 "$q" "Overall Accuracy (OA)" || echo -)
d=$(awk -v a="$ov" -v b="$bv" 'BEGIN {if (a != "-" && b != "-") printf "%+.2f", a - b; else printf "-"}')
echo "| DomainNet R2S（OA） | $bv | $ov | $d | $qv |"
echo "============================== 汇总结束 =============================="
