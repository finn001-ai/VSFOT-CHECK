#!/usr/bin/env bash
# =============================================================================
# 锁死错误结构（2026-10-18）：无标签重拟合分类头全部回到约 76（学生水平），
#   这里描述约 970 个锁死错误的形状：类对集中度、吞并类、真实类排名、子簇（标签只用于描述）
#   不训练网络，直接用 scripts/run_repr.sh 保存的快照；每个快照几分钟
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_locked.sh
# 把「汇总开始」到「汇总结束」之间贴回即可，不用下载
# =============================================================================

set -euo pipefail

PYTHON="python -u"
GPU=0
SNAP_DIR="runs/ReprDiag/both_snap/Ar2Cl_s42"     # 和 scripts/run_repr.sh 一致
STAGES=(p025 p050 p100)
RESULT_DIR="runs/locked_results/$(date +%Y_%m_%d)_Ar2Cl_both_snap"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
mkdir -p "$RESULT_DIR"
for stage in "${STAGES[@]}"; do
    snap="$SNAP_DIR/snapshot_${stage}.pt"
    out="$RESULT_DIR/$stage"
    if [ ! -f "$snap" ]; then
        echo "[错误] 找不到快照 ${snap}，请先运行 bash scripts/run_repr.sh"; exit 1
    fi
    if [ -f "$out/result.json" ]; then
        echo "==> ${stage}：已有结果，跳过"
        continue
    fi
    echo "==> 分析 ${stage}"
    mkdir -p "$out"
    $PYTHON tools/diag_locked.py --snapshot "$snap" --out "$out" --device "$GPU" 2>&1 | tee "$out/diag_log.txt"
done

$PYTHON tools/diag_locked.py --collect "$RESULT_DIR"
