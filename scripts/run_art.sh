#!/usr/bin/env bash
# =============================================================================
# 以 Art 为目标的任务诊断（2026-10-19）：cap02 与 COSMO 差距最大的是 Cl→Ar（−2.1）和 Re→Ar（−1.3）；
#   冻结源模型专家（run_src.sh）无效甚至有害（Ar→Cl −3.3），关闭。先弄清楚 Art 目标上停在哪里，再决定改什么。
#   以 Art 为目标的训练只有 1479 步（每 epoch 37 步），cap02 的曲线在第 592–777 步到峰值后略降。
#   三组都开 --log_dynamics --log_samples，并在进度 0.25 / 1.0 存快照：
#     both_cap02_dyn  Cl→Ar、Re→Ar（cap02）
#     both_ramp_dyn   Cl→Ar（底座，对照）
#   训练后自动做：共识塌缩分析（三组 + 已有的 Ar→Cl）、Cl→Ar cap02 p100 快照的表示诊断（探针 / 有效秩）和锁死错误结构。
#   每组约 25 分钟（含记录），加分析共约 1.5 小时。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_art.sh
# 把第一个「汇总开始」到最后一个「汇总结束」之间的内容都贴回即可，不用下载
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

DYN="--log_dynamics --log_samples --snapshot_at 0.25 1.0"

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"
JOBS=(
    "both_cap02_dyn|clipart:art|42|$BASE $CAP $DYN"
    "both_cap02_dyn|realworld:art|42|$BASE $CAP $DYN"
    "both_ramp_dyn|clipart:art|42|$BASE $DYN"
)

# 和 scripts/run_weight.sh 同一个目录
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_art"

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
SNAP="runs/$TRAIN_ROOT/both_cap02_dyn/Cl2Ar_s42/snapshot_p100.pt"
for tool in repr locked; do
    out="$RESULT_DIR/${tool}_Cl2Ar/p100"
    if [ ! -f "$out/result.json" ]; then
        mkdir -p "$out"
        $PYTHON tools/diag_${tool}.py --snapshot "$SNAP" --out "$out" --device "$GPU" > "$out/diag_log.txt" 2>&1 \
            || echo "[警告] diag_${tool}.py 失败，见 $out/diag_log.txt"
    fi
done
echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON tools/collapse_analysis.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR" | grep -v "^已打包" | sed -n '/^## Cl2Ar/,/^## Cl2Pr/p;/^## Re2Ar/,/^## Re2Cl/p;/^## 训练动态/,$p'
echo "============================== 汇总结束 =============================="
$PYTHON tools/diag_repr.py --collect "$RESULT_DIR/repr_Cl2Ar"
$PYTHON tools/diag_locked.py --collect "$RESULT_DIR/locked_Cl2Ar"
