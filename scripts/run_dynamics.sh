#!/usr/bin/env bash
# =============================================================================
# 训练动态诊断：为什么训练在第 11 个 epoch（第 748 步）左右就停滞
#   每个 epoch 额外记录（--log_dynamics，写到 dynamics.csv，不改变训练过程）：
#     老师（带 adapter 的 CLIP）准确率、学生 / 老师一致率、两边都错的样本数、
#     学生 / 老师的预测翻转率（和上一次评估相比改变预测的比例）、伪标签准确率、预测熵、各项损失
#   ot_knn10_both_dyn  目前最好的设置（对照：ot_knn10_both 75.98，两者准确率应完全一致）
#   ot_dyn             原方法（对照：ot 74.39）
#
# 结果写到 runs/AblationFix/<组名>/<任务>_s<种子>/，汇总表末尾附「训练动态」表
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_dynamics.sh
# 跑完下载 runs/ablation_results/<日期>.zip
#
# 耗时参考：每组比不记录时多约 10 分钟（每个 epoch 多一次全量前向）
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40              # README 命令的设置

SEEDS=(42)             # 和 AblationFix 基线相同的种子

# 实验组，格式 "组名|额外参数"；不想跑的行在前面加 # 注释掉
# 第 1 轮（2026-10-16，只跑了 Ar→Cl）：ot_strong 73.82、ot_knn10_strong 74.78 都不如对照，
# 只有 ot_knn10_both 75.98（相对 ot_knn10 +0.79）有效，所以第 2 轮只保留 both
RUN_GROUPS=(
    "ot_knn10_both_dyn|--align_mode ot --knn_k 10 --align_view both --log_dynamics"
    "ot_dyn|--align_mode ot --log_dynamics"
)

# 先只看 Ar→Cl
TASKS=(
    "art:clipart"
)

# 和 scripts/run_ablation.sh 用同一个目录，汇总时自动包含基线
TRAIN_ROOT="AblationFix"
RESULT_DIR="runs/ablation_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

abbr() {
    case "$1" in
        art) echo "Ar" ;; clipart) echo "Cl" ;; product) echo "Pr" ;; realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

# 先展开成「源域 目标域 任务名 组名 种子 | 额外参数」，并统计还需要训练几组
todo=()
n_done=0
for seed in "${SEEDS[@]}"; do
    for pair in "${TASKS[@]}"; do
        s="${pair%%:*}"
        t="${pair##*:}"
        task="$(abbr "$s")2$(abbr "$t")"
        for g in "${RUN_GROUPS[@]}"; do
            name="${g%%|*}"
            if [ -f "runs/$TRAIN_ROOT/$name/${task}_s${seed}/.done" ]; then
                echo "==> ${task} ${name} seed=${seed}：已完成，跳过"
                n_done=$((n_done + 1))
            else
                todo+=("$s $t $task $name $seed|${g#*|}")
            fi
        done
    done
done
echo "==> 已完成 ${n_done} 组，本次需要训练 ${#todo[@]} 组"

k=0
# ${todo[@]+...}：数组为空时不展开，避免 set -u 在旧版 bash 下报错
for item in ${todo[@]+"${todo[@]}"}; do
    read -r s t task name seed <<< "${item%%|*}"
    extra="${item#*|}"
    k=$((k + 1))
    dir="runs/$TRAIN_ROOT/$name/${task}_s${seed}"

    # main.py 遇到同名目录会自动加时间戳另建，导致汇总时找不到，所以提前拦下
    if [ -d "$dir" ]; then
        echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
    fi

    echo ""
    echo "################################################################"
    echo "==> 第 ${k}/${#todo[@]} 组：${task}（${s} -> ${t}），${name}，seed=${seed}"
    echo "    参数：${extra}"
    echo "################################################################"
    # $extra 不加引号，让它拆成多个参数
    $PYTHON main.py \
        --device "$GPU" \
        --name "$TRAIN_ROOT/$name/${task}_s${seed}" \
        --epochs "$EPOCHS" \
        --dataset "$DATASET" \
        --source "$s" \
        --target "$t" \
        --seed "$seed" \
        $extra
    # 训练正常结束后才写完成标记
    touch "$dir/.done"
done

echo "==> 汇总"
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
