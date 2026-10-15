#!/usr/bin/env bash
# =============================================================================
# 按论文附录（CVPR 2026 supplemental，第 4 节）的配置复现 VSFOT
#
# 和原代码默认值的区别（原代码默认值保持不变，这里通过参数传入）：
#   模型 SGD        momentum 0.9、weight decay 5e-4   （原代码：纯 SGD）
#   CLIP adapter    Adam weight decay 1e-4            （原代码：无）
#   adapter 维度    64                                 （原代码：256）
#   Office-Home     20 epoch                           （README 命令：40）
# 和论文一致、无需改动的：CLIP ViT-B/32、批大小 64、ε = 0.2、最后 3 层 adapter
#
# 论文 Ar→Cl = 75.00（3 个种子平均）；原代码配置我们得到 67.2
#
# 在 Jupyter 里运行：
#   !bash run_paper_cfg.sh
# 跑完下载 paper_cfg_results/<日期>.zip
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0

# 论文附录配置
EPOCHS=20
PAPER_ARGS="--momentum 0.9 --wd 5e-4 --adam_wd 1e-4 --adapter_dim 64"

# 任务，格式 "源域:目标域"
TASKS=(
    "art:clipart"
)

# 种子；先跑 1 个看效果，接近论文后再加 2 个（论文报 3 个种子的平均）
SEEDS=(42)

# 对齐方式，默认只跑原方法；之后可加 clip_soft clip_sharp 做消融
MODES=(ot)

# 训练输出到 process/<TRAIN_ROOT>/<模式>/<任务>_s<种子>/
TRAIN_ROOT="PaperCfg"
RESULT_DIR="paper_cfg_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")"

abbr() {
    case "$1" in
        art) echo "Ar" ;; clipart) echo "Cl" ;; product) echo "Pr" ;; realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

total=$(( ${#TASKS[@]} * ${#SEEDS[@]} * ${#MODES[@]} ))
k=0
for pair in "${TASKS[@]}"; do
    s="${pair%%:*}"
    t="${pair##*:}"
    task="$(abbr "$s")2$(abbr "$t")"
    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            k=$((k + 1))
            dir="process/$TRAIN_ROOT/$mode/${task}_s${seed}"
            if [ -f "$dir/.done" ]; then
                echo "==> 第 ${k}/${total} 组 ${task} ${mode} seed=${seed}：已完成，跳过"
                continue
            fi
            # main.py 遇到同名目录会自动加时间戳另建，导致汇总时找不到，所以提前拦下
            if [ -d "$dir" ]; then
                echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
            fi

            echo ""
            echo "################################################################"
            echo "==> 第 ${k}/${total} 组：${task}（${s} -> ${t}），模式 ${mode}，seed=${seed}"
            echo "    论文配置：${EPOCHS} epoch ${PAPER_ARGS}"
            echo "################################################################"
            # $PAPER_ARGS 不加引号，让它拆成多个参数
            $PYTHON main.py \
                --device "$GPU" \
                --name "$TRAIN_ROOT/$mode/${task}_s${seed}" \
                --epochs "$EPOCHS" \
                --dataset "$DATASET" \
                --source "$s" \
                --target "$t" \
                --seed "$seed" \
                --align_mode "$mode" \
                $PAPER_ARGS
            touch "$dir/.done"
        done
    done
done

echo "==> 汇总"
$PYTHON ablation_collect.py --process "process/$TRAIN_ROOT" --out "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
