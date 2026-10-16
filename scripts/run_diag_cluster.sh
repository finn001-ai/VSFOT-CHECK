#!/usr/bin/env bash
# =============================================================================
# 簇级命名诊断（方案 A）：学生分组、CLIP 命名，能否越过「CLIP 和学生任一判对」的并集上限
#
# 不训练，直接使用 scripts/run_diag.sh 已保存的快照（runs/DiagFix/<任务>/snapshot_*.pt）
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_diag_cluster.sh
# 跑完下载 runs/diag_cluster_results/<日期>.zip
#
# 耗时参考：每个快照几分钟（4 个 K × 3 个 k-means 种子）
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

KS="65 130 260 520"      # 簇数：越大簇越细越纯，但每簇样本越少、投票越不稳
REG=0.05                 # 簇 → 类 OT 的熵正则 ε

# 版本标签：改了命名方式后换一个标签，避免同一天的旧结果被当成「已有结果」跳过
TAG="v2_two_namers"
RESULT_DIR="runs/diag_cluster_results/$(date +%Y_%m_%d)_${TAG}"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
mkdir -p "$RESULT_DIR"

for task in "${TASKS[@]}"; do
    for stage in "${STAGES[@]}"; do
        snap="runs/$TRAIN_ROOT/$task/snapshot_${stage}.pt"
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
        $PYTHON tools/diag_cluster.py \
            --snapshot "$snap" \
            --out "$out" \
            --device "$GPU" \
            --ks $KS \
            --reg "$REG" \
            2>&1 | tee "$out/diag_log.txt"
    done
done

echo "==> 汇总"
$PYTHON tools/diag_cluster.py --collect "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
