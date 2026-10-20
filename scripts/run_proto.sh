#!/usr/bin/env bash
# =============================================================================
# 类原型修正离线检验（2026-10-20，不训练）：H1（archive/2026_10_20_h1）发现学生的缺口继承自老师的固定文本原型分类头。
#   在 H1 已存的快照上（12 个任务训练前 + 4–5 个 cap02 训练结束），从文本原型出发做无标签的原型修正（软 EM / k-means /
#   Sinkhorn 均匀 / Sinkhorn √计数，是否保留文本先验，是否中心化），看能比老师高多少、离线性探针还差多少。
#   学生瓶颈特征上也做一遍，对照之前「用自身预测重拟合线性头只有 ±0.6」。
#   需要先跑过 scripts/run_h1.sh（快照）；每个快照约 1–2 分钟（一次前向 + 很小的计算），共约 30 分钟。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_proto.sh
# 已分析的快照会自动跳过；把「汇总开始」到「汇总结束」之间贴回即可，不用下载
# =============================================================================

set -euo pipefail

PYTHON="python -u"
GPU=0
SEED=42
TASKS=(Ar2Cl Pr2Cl Re2Cl Cl2Ar Pr2Ar Re2Ar Ar2Pr Cl2Pr Re2Pr Ar2Re Cl2Re Pr2Re)
RESULT_DIR="runs/proto_results/proto"
H1_DIR="runs/h1_results/h1"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

# 分析一个快照；$1 任务 $2 阶段 $3 快照路径
analyse() {
    local task="$1" stage="$2" snap="$3" out="$RESULT_DIR/$1/$2"
    if [ ! -f "$snap" ]; then
        echo "[跳过] 找不到快照 ${snap}"; return
    fi
    if [ -f "$out/result.json" ]; then
        echo "==> ${task} ${stage}：已有结果，跳过"; return
    fi
    echo "==> 分析 ${task} ${stage}"
    mkdir -p "$out"
    $PYTHON tools/diag_proto.py --snapshot "$snap" --out "$out" --device "$GPU" > "$out/diag_log.txt" 2>&1 \
        && tail -n 2 "$out/diag_log.txt" \
        || echo "[警告] ${task} ${stage} 分析失败，见 ${out}/diag_log.txt"
}

for task in "${TASKS[@]}"; do
    analyse "$task" p000 "runs/H1Probe/p000/${task}_s${SEED}/snapshot_p000.pt"
done
for task in Ar2Pr Ar2Re; do
    analyse "$task" p100 "runs/H1Probe/cap02_snap/${task}_s${SEED}/snapshot_p100.pt"
done
for task in Cl2Ar Re2Ar Ar2Cl; do
    analyse "$task" p100 "runs/WeightSched/both_cap02_dyn/${task}_s${SEED}/snapshot_p100.pt"
done

$PYTHON tools/diag_proto.py --collect "$RESULT_DIR" --h1_root "$H1_DIR" || echo "[警告] 汇总失败"
