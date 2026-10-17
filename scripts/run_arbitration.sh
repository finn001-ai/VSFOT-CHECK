#!/usr/bin/env bash
# =============================================================================
# 分歧裁决分析（不训练，只用 CPU，几秒钟）
#   读取 scripts/run_collapse.sh 保存的逐样本预测 runs/AblationFix/*/*/samples.npz，
#   检验学生和老师意见不一致时，熵 / 预测稳定性 / 源模型投票等无标签信号能否分辨谁对
#   （见 archive/2026_10_17_collapse）
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_arbitration.sh
# 跑完下载 runs/arbitration_results/<日期>.zip（或直接复制打印出来的表格）
# =============================================================================

set -euo pipefail

TRAIN_ROOT="AblationFix"
RESULT_DIR="runs/arbitration_results/$(date +%Y_%m_%d)"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
python -u tools/arbitration_analysis.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
