#!/usr/bin/env bash
# =============================================================================
# Ar→Cl 对照测试：类名修复（classname.txt）之后，重新跑两组
#
#   A  code_readme  原公开代码公式（--formula code）+ README 配置（40 epoch）
#                   对比之前的 67.2%  → 类名错位（第 58-63 类）影响有多大
#   B  paper_cfg    论文公式（--formula paper）+ 附录配置（20 epoch，momentum 等）
#                   对比之前崩溃的 9.3% → 崩溃是不是类名错位造成的
#
# 前提：dataset/OfficeHomeDataset/classname.txt 已链接（bash link_data.sh，或
#   ln -sfn /hyperai/input/input0/classname.txt dataset/OfficeHomeDataset/classname.txt）
#
# 在 Jupyter 里运行：
#   !bash run_ac.sh
# 耗时：A 约 30 分钟，B 约 15 分钟。跑完下载 ac_results/<日期>.zip
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
PYTHON="python -u"
GPU=0
SEED=42
SRC="art"
TGT="clipart"
TASK="Ar2Cl"

# 每组格式 "组名|额外参数"；不想跑的组在前面加 # 注释掉
RUN_GROUPS=(
    "code_readme|--formula code --epochs 40"
    "paper_cfg|--formula paper --epochs 20 --momentum 0.9 --wd 5e-4 --adam_wd 1e-4 --adapter_dim 64"
)

# 训练输出到 process/<TRAIN_ROOT>/<组名>/<任务>_s<种子>/
TRAIN_ROOT="ClassFix"
RESULT_DIR="ac_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")"

if [ ! -f "dataset/OfficeHomeDataset/classname.txt" ]; then
    echo "[错误] 找不到 dataset/OfficeHomeDataset/classname.txt，请先执行 bash link_data.sh"; exit 1
fi

k=0
for g in "${RUN_GROUPS[@]}"; do
    name="${g%%|*}"
    extra="${g#*|}"
    k=$((k + 1))
    dir="process/$TRAIN_ROOT/$name/${TASK}_s${SEED}"

    if [ -f "$dir/.done" ]; then
        echo "==> 第 ${k}/${#RUN_GROUPS[@]} 组 ${name}：已完成，跳过"
        continue
    fi
    # main.py 遇到同名目录会自动加时间戳另建，导致汇总时找不到，所以提前拦下
    if [ -d "$dir" ]; then
        echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
    fi

    echo ""
    echo "################################################################"
    echo "==> 第 ${k}/${#RUN_GROUPS[@]} 组：${name}（${TASK}，seed=${SEED}）"
    echo "    参数：${extra}"
    echo "################################################################"
    # $extra 不加引号，让它拆成多个参数
    $PYTHON main.py \
        --device "$GPU" \
        --name "$TRAIN_ROOT/$name/${TASK}_s${SEED}" \
        --dataset OfficeHomeDataset \
        --source "$SRC" \
        --target "$TGT" \
        --seed "$SEED" \
        --align_mode ot \
        $extra
    touch "$dir/.done"
done

echo "==> 汇总"
$PYTHON ablation_collect.py --process "process/$TRAIN_ROOT" --out "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
