#!/usr/bin/env bash
# =============================================================================
# 对齐权重曲线（2026-10-18）：老师只在前约 7.5% 的训练里比学生准，原方法的对齐权重 min(1, 2·进度) 却在那之后才加满；
#   锁死样本 220 → 645 → 854 → 970（p000 / 025 / 050 / 100），大部分锁死发生在权重上升的阶段。
#   离线诊断（archive/2026_10_18_repr）说明：后期从当前预测里已经提不出新信息（近邻 / 重拟合头 / 类边际都 ≈ 学生），
#   所以改为在源头少继承老师的系统性错误：
#     both_const   权重一直为 1（老师更准的早期全力对齐，之后不再额外加大）
#     both_cap02   权重 min(0.2, 2·进度)：进度 0.1（交叉点附近）后保持 0.2
#   底座 ot_knn10_both（75.98），不开诊断记录，每组约 35 分钟
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_weight.sh
# 把「汇总开始」到「汇总结束」之间贴回即可，不用下载
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

BASE="--align_mode ot --knn_k 10 --align_view both"

# 实验组，格式 "组名|额外参数"；不想跑的行在前面加 # 注释掉
RUN_GROUPS=(
    "both_const|$BASE --align_weight const"
    "both_cap02|$BASE --align_weight cap --align_cap 0.2"
)

# 先只看 Ar→Cl
TASKS=(
    "art:clipart"
)

# 单独的目录：汇总只含这一轮（底座 ot_knn10_both = 75.98 见 archive/2026_10_16_strong）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)"

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
echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR" | grep -v "^已打包"
echo "============================== 汇总结束 =============================="
