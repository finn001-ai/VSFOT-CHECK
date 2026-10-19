#!/usr/bin/env bash
# =============================================================================
# cap02 的多种子结果（2026-10-19）：Office-Home 12 个任务 seed 42 上 cap02 平均 86.21（底座 85.57，+0.65，12 / 12 个任务都提升），
#   COSMO 86.4 是 3 个种子的均值，单种子差 0.19 在种子波动范围内，必须补种子才能比较。
#   这一轮：12 个任务 × cap02 × seed 0、1（Ar→Cl 已有，自动跳过），共 22 组；
#   之后如需底座的 3 种子对照，把下面 BASE_SEEDS 那一行取消注释（再加 22 组）。
#   每组 18–57 分钟（Re 为目标域的任务最慢，图片大、数据加载占主导），22 组约 12 小时；中断后重跑会跳过已完成的组。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_seeds.sh
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
DOMAINS=(art clipart product realworld)
for seed in 0 1; do
    for src in "${DOMAINS[@]}"; do
        for tgt in "${DOMAINS[@]}"; do
            [ "$src" = "$tgt" ] && continue
            JOBS+=("both_cap02|$src:$tgt|$seed|$BASE $CAP")
            # BASE_SEEDS：底座的 3 种子对照，需要时取消注释
            # JOBS+=("both_ramp|$src:$tgt|$seed|$BASE")
        done
    done
done

# 和 scripts/run_weight.sh 同一个目录
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_seeds"

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
