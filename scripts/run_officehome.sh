#!/usr/bin/env bash
# =============================================================================
# 对齐权重上限：消融 + 机制 + Office-Home 全部 12 个任务（2026-10-18）
#   run_cap.sh 已确认：cap02 在 Ar→Cl 3 个种子 77.17 ± 0.20（底座 76.04 ± 0.13），Pr→Cl 77.21、Re→Cl 77.04，三个任务都 +1.0；
#   上限扫描 0.1 / 0.2 / 0.3 / 0.5 / 1.0（底座）= 76.98 / 77.00 / 76.64 / 76.16 / 75.98，越小越好、0.1–0.2 持平。
#   这一轮按优先顺序：
#     1. 消融（Ar→Cl）：cap 0.05、cap 0（学生完全不对齐老师，只有 MGVA + SimSiam + IM + 近邻平滑不起作用）
#        —— 回答「是限制拉力有用，还是对齐本身就该去掉」
#     2. 机制（Ar→Cl）：cap02 加 --log_dynamics --log_samples，和 AblationFix/both_samples 对照锁死 / 丢失 / 并集
#        （记录比普通训练多约 40%）
#     3. Office-Home 12 个任务 × {底座 ramp, cap02}，seed 42（已完成的自动跳过；Ar/Pr/Re→Cl 的 cap02 已有）
#   每组约 26 分钟（本机），共约 25 组；不想跑的行在前面加 # 注释掉。Cl→X 任务需要源模型 Source_clipart。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_officehome.sh
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
JOBS=(
    "both_cap00|art:clipart|42|$BASE --align_weight cap --align_cap 0"
    "both_cap005|art:clipart|42|$BASE --align_weight cap --align_cap 0.05"
    "both_cap02_dyn|art:clipart|42|$BASE $CAP --log_dynamics --log_samples"
)
# Office-Home 12 个任务：先 cap02 再底座，按源域顺序
DOMAINS=(art clipart product realworld)
for src in "${DOMAINS[@]}"; do
    for tgt in "${DOMAINS[@]}"; do
        [ "$src" = "$tgt" ] && continue
        JOBS+=("both_cap02|$src:$tgt|42|$BASE $CAP")
        JOBS+=("both_ramp|$src:$tgt|42|$BASE")
    done
done

# 和 scripts/run_weight.sh 同一个目录
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_officehome"

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
$PYTHON tools/collapse_analysis.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR" | grep -v "^已打包"
echo "============================== 汇总结束 =============================="
