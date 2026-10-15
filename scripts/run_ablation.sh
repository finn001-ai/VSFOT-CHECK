#!/usr/bin/env bash
# =============================================================================
# OT 消融：完整训练，对比对齐损失用的计划 gamma1
#   ot          原方法（Sinkhorn 计划）
#   clip_soft   不做 OT，直接用 CLIP 软标签
#   clip_hard   不做 OT，直接用 CLIP argmax 的 one-hot
#   clip_sharp  和 OT 同一个核（锐化），只做列归一化，不做类别边际投影
#               ot - clip_sharp = 边际投影的作用；clip_sharp - clip_soft = 锐化的作用
# 其余（类权重、SimSiam、IM 损失、CLIP 反向蒸馏、学习率等）完全不变
#
# 前提：已执行 scripts/link_data.sh，源模型已训练好（process/Source_home/...）
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_ablation.sh
# 跑完下载 ablation_results/<日期>.zip
#
# 已完成的组（有 .done 标记）会自动跳过，可以放心重跑；中断的组按提示删除目录后重跑。
# 耗时参考：目标域为 Clipart 的任务每组约 30 分钟，Pr2Re 每组约 85 分钟
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40              # README 命令的设置

# 随机种子；每个种子把下面所有组各跑一遍，汇总时取均值 ± 标准差
SEEDS=(42)

# 任务和要跑的模式，格式 "源域:目标域:模式1,模式2,..."
# 不想跑的行在前面加 # 注释掉；想加模式直接在逗号后追加
RUNS=(
    "art:clipart:ot,clip_soft,clip_sharp"
    "product:clipart:ot,clip_soft,clip_sharp"
    "realworld:clipart:ot,clip_soft,clip_sharp"
    # "product:realworld:ot,clip_soft,clip_sharp"     # 简单任务对照，每组约 85 分钟
)

# 训练输出到 process/<TRAIN_ROOT>/<模式>/<任务>_s<种子>/
# AblationFix = 类名修复后的消融；旧的 process/Ablation/ 是修复前的结果，保留做对比
TRAIN_ROOT="AblationFix"
RESULT_DIR="ablation_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
# 切换到项目根目录（本脚本在 scripts/ 下）
cd "$(dirname "$0")/.."

abbr() {
    case "$1" in
        art) echo "Ar" ;; clipart) echo "Cl" ;; product) echo "Pr" ;; realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

# 先展开成「源域 目标域 任务名 模式 种子」的列表，并统计还需要训练几组
todo=()
n_done=0
for seed in "${SEEDS[@]}"; do
    for run in "${RUNS[@]}"; do
        IFS=":" read -r s t modes <<< "$run"
        task="$(abbr "$s")2$(abbr "$t")"
        IFS="," read -r -a mode_list <<< "$modes"
        for mode in "${mode_list[@]}"; do
            if [ -f "process/$TRAIN_ROOT/$mode/${task}_s${seed}/.done" ]; then
                echo "==> ${task} ${mode} seed=${seed}：已完成，跳过"
                n_done=$((n_done + 1))
            else
                todo+=("$s $t $task $mode $seed")
            fi
        done
    done
done
echo "==> 已完成 ${n_done} 组，本次需要训练 ${#todo[@]} 组"

k=0
# ${todo[@]+...}：数组为空时不展开，避免 set -u 在旧版 bash 下报错
for item in ${todo[@]+"${todo[@]}"}; do
    read -r s t task mode seed <<< "$item"
    k=$((k + 1))
    dir="process/$TRAIN_ROOT/$mode/${task}_s${seed}"

    # main.py 遇到同名目录会自动加时间戳另建，导致汇总时找不到，所以提前拦下
    if [ -d "$dir" ]; then
        echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
    fi

    echo ""
    echo "################################################################"
    echo "==> 第 ${k}/${#todo[@]} 组：${task}（${s} -> ${t}），模式 ${mode}，seed=${seed}"
    echo "################################################################"
    $PYTHON main.py \
        --device "$GPU" \
        --name "$TRAIN_ROOT/$mode/${task}_s${seed}" \
        --epochs "$EPOCHS" \
        --dataset "$DATASET" \
        --source "$s" \
        --target "$t" \
        --seed "$seed" \
        --align_mode "$mode"
    # 训练正常结束后才写完成标记
    touch "$dir/.done"
done

echo "==> 汇总"
$PYTHON tools/ablation_collect.py --process "process/$TRAIN_ROOT" --out "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
