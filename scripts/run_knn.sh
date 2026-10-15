#!/usr/bin/env bash
# =============================================================================
# 近邻平滑实验：在学生特征空间用 k 近邻修正伪标签（静态诊断见 archive/2026_10_16_diag_struct）
#   ot_knn10       学生的对齐目标 = 近邻平滑后的 CLIP 概率（--knn_k 10）
#   ot_knn10_mgva  同上，且 MGVA 中老师学的目标 = 近邻平滑后的学生概率（--knn_k 10 --knn_mgva）
# 两组都基于 --align_mode ot，其余与公开代码完全一致
#
# 结果写到 runs/AblationFix/<组名>/<任务>_s<种子>/，和已有的 ot / clip_soft 基线放在一起汇总
#
# 前提：已执行 scripts/link_data.sh，源模型在 runs/Source_home/
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_knn.sh
# 跑完下载 runs/ablation_results/<日期>.zip（汇总表含「相对 ot」一列）
#
# 已完成的组（有 .done 标记）会自动跳过；耗时参考：每组约 22 分钟
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
RUN_GROUPS=(
    "ot_knn10|--align_mode ot --knn_k 10"
    "ot_knn10_mgva|--align_mode ot --knn_k 10 --knn_mgva"
)

# 任务，格式 "源域:目标域"
TASKS=(
    "art:clipart"
    "product:clipart"
    "realworld:clipart"
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
