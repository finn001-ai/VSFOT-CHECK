#!/usr/bin/env bash
# =============================================================================
# 冻结源模型作为第二个专家（2026-10-19）：cap02 在 12 个任务上平均 86.21，COSMO 86.4；
#   以 Clipart 为目标的 3 个任务我们都更高（+0.6 / +1.1 / +1.5），其余多数更低，差距最大的是以 Art 为目标（Cl→Ar −2.1、Re→Ar −1.3）
#   和以 Art 为源（Ar→Pr、Ar→Re −0.7）。COSMO / ProDe 都用冻结的源模型和 CLIP 组成按熵加权的专家乘积，VSFOT 训练开始后就不再用源模型。
#   --src_expert：对齐目标里的 CLIP 概率（自身和近邻的）换成 CLIP × 冻结源模型（按熵加权），其余同 cap02。
#     cap02_src      全程使用
#     cap02_src050   只在前一半训练使用（之后学生已远超源模型）
#   先在差距大的 3 个任务 + Ar→Cl（防止原本最好的任务变差）上筛选；对照 cap02 的已有结果在同一个目录，汇总表里并排显示。
#   每组 18–26 分钟，共 8 组约 3 小时。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_src.sh
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

BASE="--align_mode ot --knn_k 10 --align_view both"

CAP="--align_weight cap --align_cap 0.2"

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"
JOBS=()
for g in "cap02_src|--src_expert" "cap02_src050|--src_expert --src_expert_until 0.5"; do
    for pair in clipart:art realworld:art art:product art:clipart; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_src"

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
for job in "${JOBS[@]}"; do
    IFS='|' read -r name pair seed extra <<< "$job"
    s="${pair%%:*}"
    t="${pair##*:}"
    task="$(abbr "$s")2$(abbr "$t")"
    if [ -f "runs/$TRAIN_ROOT/$name/${task}_s${seed}/.done" ]; then
        echo "==> ${task} ${name} seed=${seed}：已完成，跳过"
        n_done=$((n_done + 1))
    else
        todo+=("$s $t $task $name $seed|$extra")
    fi
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
