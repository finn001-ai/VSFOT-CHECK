#!/usr/bin/env bash
# =============================================================================
# 目标域结构诊断：用学生模型特征空间的结构（近邻 / 原型 / gamma2）修正 CLIP 伪标签，能否超过 CLIP
#
# 不训练，直接使用 scripts/run_diag.sh 已保存的快照（runs/process/DiagFix/<任务>/snapshot_*.pt）
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_diag_struct.sh
# 跑完下载 runs/diag_struct_results/<日期>.zip
#
# 耗时参考：每个快照几分钟
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
PYTHON="python -u"
GPU=0

# 和 scripts/run_diag.sh 保持一致
TRAIN_ROOT="DiagFix"
STAGES=(p000 p025 p050)
TASKS=(Ar2Cl Pr2Re)

DIAG_BATCH=64            # gamma2 分批大小（同训练）
DIAG_SEEDS="0 1 2"       # gamma2 随机分批的种子

RESULT_DIR="runs/diag_struct_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
mkdir -p "$RESULT_DIR"

for task in "${TASKS[@]}"; do
    for stage in "${STAGES[@]}"; do
        snap="runs/process/$TRAIN_ROOT/$task/snapshot_${stage}.pt"
        out="$RESULT_DIR/$task/$stage"
        if [ ! -f "$snap" ]; then
            echo "[错误] 找不到快照 ${snap}，请先运行 bash scripts/run_diag.sh"; exit 1
        fi
        if [ -f "$out/summary.json" ]; then
            echo "==> ${task} ${stage}：已有结果，跳过"
            continue
        fi
        echo "==> ${task} ${stage}"
        mkdir -p "$out"
        $PYTHON tools/diag_struct.py \
            --snapshot "$snap" \
            --out "$out" \
            --device "$GPU" \
            --batch_size "$DIAG_BATCH" \
            --seeds $DIAG_SEEDS \
            2>&1 | tee "$out/diag_log.txt"
    done
done

echo "==> 汇总"
$PYTHON tools/diag_struct.py --collect "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
