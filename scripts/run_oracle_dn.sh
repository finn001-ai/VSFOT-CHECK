#!/usr/bin/env bash
# =============================================================================
# 类先验上界：DomainNet（2026-10-23）：--clsw_mode oracle，类均衡的目标分布换成目标域真实类频率（用了标签，诊断，不是方法）
#   1/频率重加权把学生的预测分布推向均匀：Office-Home Art 目标上小类被多预测 15 ~ 19%（输 COSMO 最多的目标域），
#   VisDA-C 上最大的类 car（10401 张，占 18.8%）只被预测出 0.67 倍，1960 张判成 truck，car 准确率 62%。
#   之前的无标签先验（CLIP 零样本平均概率，archive/2026_10_20_ab）−1.33，因为 CLIP 的先验本身偏 8 ~ 19 倍。
#   run_oracle.sh：Cl→Ar +0.96（84.73，COSMO 85.7），VisDA AA 最后一次 +0.82（89.46，COSMO 91.4）、OA +2.7。
#   DomainNet R→S 小类被多预测 1.60 倍（Office-Home 最多 1.28），指标是 OA，上界可能更大：
#     R→S，cap02 + tstrong，10 epoch，对照 run_domainnet.sh 的 cap02_tstrong_e10（77.95，COSMO 79.8）
#
#   !bash scripts/run_oracle_dn.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_oracle_dn.sh
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
    "DomainNet/cap02_tstrong_oracle_e10/R2S_s42|DomainNet|real|sketch|--epochs 10 --arch resnet50"
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

last5() { grep -o "$2: [0-9.]*" "$1" | tail -n +2 | tail -5 | awk '{s+=$NF} END {if (NR) printf "%.2f", s/NR*100; else printf "-"}'; }
echo "============================== 汇总开始（把这一段完整贴回）=============================="
echo "## 类先验上界：DomainNet R→S（10 epoch，seed 42；oracle = 真实类频率作为均衡目标，用了标签）"
echo ""
echo "| 指标 | 对照 cap02_tstrong_e10 | oracle | 差 | COSMO |"
echo "|---|---|---|---|---|"
for m in "Overall Accuracy (OA)" "Average Accuracy (AA)"; do
    b="runs/DomainNet/cap02_tstrong_e10/R2S_s42/Detail.txt"; o="runs/DomainNet/cap02_tstrong_oracle_e10/R2S_s42/Detail.txt"
    bv=$([ -f "$b" ] && last5 "$b" "$m" || echo -); ov=$([ -f "$o" ] && last5 "$o" "$m" || echo -)
    d=$(awk -v a="$ov" -v b="$bv" 'BEGIN {if (a != "-" && b != "-") printf "%+.2f", a - b; else printf "-"}')
    echo "| ${m%% *} | $bv | $ov | $d | $([ "${m%% *}" = Overall ] && echo 79.8 || echo -) |"
done
echo "============================== 汇总结束 =============================="
