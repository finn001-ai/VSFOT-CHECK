#!/usr/bin/env bash
# =============================================================================
# 交叉点检测（不训练，只用 CPU，几秒钟）
#   读取 scripts/run_collapse.sh 保存的逐样本预测 runs/AblationFix/*/*/samples.npz，
#   检验不用标签能否判断「当前阶段学生和老师谁更准」，以及估出的交叉点离真实的约第 204 步有多远
#   （见 archive/2026_10_17_collapse；逐样本裁决见 scripts/run_arbitration.sh）
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_crossover.sh
# 把「汇总开始」到「汇总结束」之间贴回即可，不用下载
# =============================================================================

set -euo pipefail

TRAIN_ROOT="AblationFix"
RESULT_DIR="runs/crossover_results/$(date +%Y_%m_%d)"

cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
python -u tools/crossover_analysis.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
