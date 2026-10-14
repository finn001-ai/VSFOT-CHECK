#!/usr/bin/env bash
# =============================================================================
# OT 消融：完整训练，对比对齐损失用的计划
#   ot         原方法（Sinkhorn 计划）
#   clip_soft  不做 OT，直接用 CLIP 软标签当计划
#   clip_hard  不做 OT，直接用 CLIP argmax 的 one-hot 当计划
# 其余（类权重、SimSiam、IM 损失、CLIP 反向蒸馏、学习率等）完全不变
#
# 前提：源模型已训练好（process/Source_home/...）
#
# 在 Jupyter 里运行：
#   !bash run_ablation.sh
# 跑完下载 ablation_results/<日期>.zip
#
# 耗时参考：上次 Ar2Cl 跑一半约 15 分钟、Pr2Re 约 40 分钟，
# 所以完整训练 2 个任务 × 3 种模式约 5～6 小时
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40

# 要对比的模式；不想跑的在前面加 # 注释掉
MODES=(
    "ot"
    "clip_soft"
    "clip_hard"
)

# 任务，格式 "源域:目标域"
TASKS=(
    "art:clipart"
    "product:realworld"
)

# 训练输出到 process/<TRAIN_ROOT>/<模式>/<任务>/
TRAIN_ROOT="Ablation"
RESULT_DIR="ablation_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")"

abbr() {
    case "$1" in
        art) echo "Ar" ;; clipart) echo "Cl" ;; product) echo "Pr" ;; realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

for pair in "${TASKS[@]}"; do
    s="${pair%%:*}"
    t="${pair##*:}"
    task="$(abbr "$s")2$(abbr "$t")"
    for mode in "${MODES[@]}"; do
        dir="process/$TRAIN_ROOT/$mode/$task"

        # 训练正常结束后才写 .done 标记；有标记就跳过
        if [ -f "$dir/.done" ]; then
            echo "==> ${task} ${mode}：已完成，跳过"
            continue
        fi
        # main.py 遇到同名目录会自动加时间戳另建，导致汇总时找不到，所以提前拦下
        if [ -d "$dir" ]; then
            echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
        fi

        echo "==> ${task} ${mode}：开始训练"
        $PYTHON main.py \
            --device "$GPU" \
            --name "$TRAIN_ROOT/$mode/$task" \
            --epochs "$EPOCHS" \
            --dataset "$DATASET" \
            --source "$s" \
            --target "$t" \
            --align_mode "$mode"
        touch "$dir/.done"
    done
done

echo "==> 汇总"
$PYTHON ablation_collect.py --process "process/$TRAIN_ROOT" --out "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
