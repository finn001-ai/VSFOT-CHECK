#!/usr/bin/env bash
# =============================================================================
# 2026-11-02 论文诊断图用的重跑（不是新实验）：与主表 / 模块消融完全相同的命令（seed 42），
#   只多开 --log_dynamics --log_samples（记录前后恢复随机数状态，不影响训练），并保存 final.pt
#   用途：t-SNE、训练动态（对齐目标准确率、师生一致 / 共同错误）、VisDA 各类准确率与预测分布
#   seed 42 的「完整」行在 --save_final 之前跑的，没有 final.pt；消融行重跑是为了拿到同口径的训练动态
#   核对：每个重跑的最终 OA / AA 必须和原目录逐位相同（汇总里打印两者）
#   OH 只跑 Ar2Cl（最难、各模块差距最大）；VisDA 跑 完整 / −A / −B / −C / VSFOT
#
#   !bash scripts/run_paper_dyn.sh          !SUMMARY_ONLY=1 bash scripts/run_paper_dyn.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
DYN="--log_dynamics --log_samples"

A="--knn_k 10 --align_view both --align_weight cap --align_cap 0.2"
OH0="--epochs 40 --align_mode ot --s_ema 0.5"
VD0="--arch resnet101 --workers 12 --epochs 15 --batch_size 32 --align_mode ot --s_ema 0.5"
VD_B="--t_strong --consensus teacher"; VD_C="--clsw_mode self --align_norm_c 65"

# 格式 "新目录|原目录（核对用）|数据集|源域|目标域|参数"
JOBS=(
    "PaperDyn/oh_full/Ar2Cl_s42|WeightSched/cap02_tstrong_sema/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH0 $A --t_strong"
    "PaperDyn/oh_minusA/Ar2Cl_s42|WeightSched/mod_minusA/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH0 --t_strong"
    "PaperDyn/oh_minusB/Ar2Cl_s42|WeightSched/mod_minusB/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH0 $A"
    "PaperDyn/oh_vsfot/Ar2Cl_s42|WeightSched/mod_vsfot/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH0"
    "PaperDyn/vd_full/S2R_s42|VisDA/norm65_self_ct_e15_b32_sema/S2R_s42|VisDA|synthesis|real|$VD0 $A $VD_B $VD_C"
    "PaperDyn/vd_minusB/S2R_s42|VisDA/mod_minusB/S2R_s42|VisDA|synthesis|real|$VD0 $A $VD_C"
    "PaperDyn/vd_minusA/S2R_s42|VisDA/mod_minusA/S2R_s42|VisDA|synthesis|real|$VD0 $VD_B $VD_C"
    "PaperDyn/vd_minusC/S2R_s42|VisDA/mod_minusC/S2R_s42|VisDA|synthesis|real|$VD0 $A $VD_B"
    "PaperDyn/vd_vsfot/S2R_s42|VisDA/mod_vsfot/S2R_s42|VisDA|synthesis|real|$VD0"
)

if [ "$SUMMARY_ONLY" != "1" ]; then
    mkdir -p runs/paper_dyn_logs
    echo "机器 $(hostname)，GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)，torch $($PYTHON -c 'import torch; print(torch.__version__)')，commit $(git log -1 --format=%h)"
    T0=$(date +%s)
    for item in "${JOBS[@]}"; do
        IFS="|" read -r dir ref ds s t extra <<< "$item"
        if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
        if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
        echo "==> 开始 $dir（$(date +%H:%M)）  参数：$extra $DYN"
        # $extra 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个
        if $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 $extra $DYN \
            > "runs/paper_dyn_logs/$(echo "$dir" | tr / _).log" 2>&1; then
            touch "runs/$dir/.done"; echo "==> 完成 $dir"
        else
            echo "[错误] $dir 训练失败，见 runs/paper_dyn_logs/"
        fi
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
# 逐位核对：重跑与原目录最后一次评估的 OA / AA（Detail.txt）和 EMA 曲线最后一行（ModellineAcc_ema.txt）
for item in "${JOBS[@]}"; do
    IFS="|" read -r dir ref ds s t extra <<< "$item"
    [ -f "runs/$dir/.done" ] || { echo "$dir：未完成"; continue; }
    new=$(grep -a "Overall Accuracy\|Average Accuracy" "runs/$dir/Detail.txt" | tail -2 | tr -s " \n" " ")
    old=$(grep -a "Overall Accuracy\|Average Accuracy" "runs/$ref/Detail.txt" | tail -2 | tr -s " \n" " ")
    ne=$(tail -1 "runs/$dir/ModellineAcc_ema.txt" 2>/dev/null | md5sum | cut -c1-8); oe=$(tail -1 "runs/$ref/ModellineAcc_ema.txt" 2>/dev/null | md5sum | cut -c1-8)
    [ "$new" = "$old" ] && [ "$ne" = "$oe" ] && v="逐位一致" || v="不一致！原 $old / EMA $oe；新 EMA $ne"
    echo "$dir：$new  [$v]"
done
echo "============================== 汇总结束 =============================="
