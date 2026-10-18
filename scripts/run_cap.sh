#!/usr/bin/env bash
# =============================================================================
# 对齐权重上限的验证（2026-10-18）：both_cap02（对齐权重 min(0.2, 2·进度)）在 Ar→Cl 上 77.00，底座 75.98（+1.02），
#   而且第 748 步之后曲线还在涨（76.0 → 77.4）；both_const（权重一直为 1）75.57，峰值后下降。
#   这一轮确认它不是单任务 / 单种子 / 单个上限值的偶然：
#     1. 泛化：Pr→Cl、Re→Cl 上的 cap02（底座 76.20 / 76.04，见 archive/2026_10_16_strong）
#     2. 上限扫描：Ar→Cl 上 cap 0.1 / 0.3 / 0.5
#     3. 种子：Ar→Cl 上 cap02 与底座（ramp）各加 seed 0、1
#     4. 底座 seed 42 在本机复现（应为 75.98；同时作为汇总表里的对照行）
#   按上面的优先顺序训练，每组约 35 分钟，共 10 组；不想跑的行在前面加 # 注释掉；已完成的组自动跳过。
#   和 scripts/run_weight.sh 用同一个目录，汇总里自动包含 both_cap02 / both_const 的结果。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_cap.sh
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

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"
JOBS=(
    "both_cap02|product:clipart|42|$BASE --align_weight cap --align_cap 0.2"
    "both_cap02|realworld:clipart|42|$BASE --align_weight cap --align_cap 0.2"
    "both_cap01|art:clipart|42|$BASE --align_weight cap --align_cap 0.1"
    "both_cap03|art:clipart|42|$BASE --align_weight cap --align_cap 0.3"
    "both_cap05|art:clipart|42|$BASE --align_weight cap --align_cap 0.5"
    "both_cap02|art:clipart|0|$BASE --align_weight cap --align_cap 0.2"
    "both_ramp|art:clipart|0|$BASE"
    "both_cap02|art:clipart|1|$BASE --align_weight cap --align_cap 0.2"
    "both_ramp|art:clipart|1|$BASE"
    "both_ramp|art:clipart|42|$BASE"
)

# 和 scripts/run_weight.sh 同一个目录
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_cap"

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
