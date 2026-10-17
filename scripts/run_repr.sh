#!/usr/bin/env bash
# =============================================================================
# 表示诊断（2026-10-17）：后期停滞时学生的表示有没有被同化；超出老师的信息在哪一层、用什么距离能拿到
#   1. 训练一次 ot_knn10_both（不开 --log_dynamics，只在进度 0 / 0.25 / 0.5 / 0.75 / 1.0 存快照）
#      存快照不影响训练，最后 5 次评估均值应和 ot_knn10_both 的 75.98 相同
#   2. tools/diag_repr.py 对每个快照算：各层（layer3 / 2048 / 瓶颈 256 / CLIP）的有效秩、线性探针、
#      近邻标签准确率（普通 / 互近邻 / CSLS）、可直接用于方法的无标签近邻信号
#   3. 汇总成一个 summary.md 并完整打印 —— 把「汇总开始」到「汇总结束」之间贴回即可，不用下载
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_repr.sh
#
# 耗时参考：训练约 35 分钟 + 每个快照几分钟；训练已完成 / 快照已分析的会自动跳过
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40
SEED=42
SRC="art"; TGT="clipart"; TASK="Ar2Cl"

NAME="both_snap"
EXTRA="--align_mode ot --knn_k 10 --align_view both --snapshot_at 0 0.25 0.5 0.75 1.0"
STAGES=(p000 p025 p050 p075 p100)

TRAIN_ROOT="ReprDiag"
RESULT_DIR="runs/repr_results/$(date +%Y_%m_%d)_${TASK}_${NAME}"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
dir="runs/$TRAIN_ROOT/$NAME/${TASK}_s${SEED}"

if [ -f "$dir/.done" ]; then
    echo "==> 训练已完成，跳过：$dir"
else
    # main.py 遇到同名目录会自动加时间戳另建，导致找不到快照，所以提前拦下
    if [ -d "$dir" ]; then
        echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
    fi
    echo "==> 训练：${TASK} ${NAME}  参数：${EXTRA}"
    $PYTHON main.py \
        --device "$GPU" \
        --name "$TRAIN_ROOT/$NAME/${TASK}_s${SEED}" \
        --epochs "$EPOCHS" \
        --dataset "$DATASET" \
        --source "$SRC" \
        --target "$TGT" \
        --seed "$SEED" \
        $EXTRA
    touch "$dir/.done"
fi

mkdir -p "$RESULT_DIR"
for stage in "${STAGES[@]}"; do
    snap="$dir/snapshot_${stage}.pt"
    out="$RESULT_DIR/$stage"
    if [ ! -f "$snap" ]; then
        echo "[错误] 找不到快照 ${snap}"; exit 1
    fi
    if [ -f "$out/result.json" ]; then
        echo "==> ${stage}：已有结果，跳过"
        continue
    fi
    echo "==> 分析 ${stage}"
    mkdir -p "$out"
    $PYTHON tools/diag_repr.py --snapshot "$snap" --out "$out" --device "$GPU" 2>&1 | tee "$out/diag_log.txt"
done

$PYTHON tools/diag_repr.py --collect "$RESULT_DIR" --run_log "$dir/run_log.txt"
