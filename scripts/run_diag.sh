#!/usr/bin/env bash
# =============================================================================
# 实验 0：OT 静态诊断
#   1. 训练到 1/2 进度，在 0、1/4、1/2 处保存快照（学习率调度仍按完整训练计算）
#   2. 对每个快照运行 diag_ot.py
#   3. 汇总结果并打包成 zip，下载后发给我分析
#
# 前提：已经执行过 bash scripts/link_data.sh，且源模型已训练好（runs/process/Source_home/...）
#
# 在 Jupyter 里运行（输出会实时显示在单元格里）：
#   !bash scripts/run_diag.sh
# =============================================================================

# 任一命令出错就立即退出；使用未定义变量也会报错；管道中任一环节失败视为失败
set -euo pipefail

# -----------------------------------------------------------------------------
# 一、配置
# -----------------------------------------------------------------------------

DATASET="OfficeHomeDataset"

# Python 解释器；-u 让输出不经缓冲，Jupyter 里能实时看到
PYTHON="python -u"

GPU=0

# 训练配置（和正式训练保持一致；批大小用 main.py 的默认值 64）
EPOCHS=40

# 保存快照的训练进度，以及训练到哪里停止
SNAPSHOT_AT="0 0.25 0.5"
STOP_AT=0.5

# 快照文件名后缀，必须和 SNAPSHOT_AT 对应（main.py 按 进度×100 补齐三位命名）
STAGES=(p000 p025 p050)

# 诊断配置
DIAG_BATCH=64
DIAG_SEEDS="0 1 2"        # 每个种子把目标域打乱一遍、遍历一整个 epoch
DIAG_SCALES="50 100 200"  # CLIP logit 尺度网格（代码默认 100）
DIAG_REGS="0.1 0.2 0.4"   # Sinkhorn ε 网格（代码默认 0.2）

# 要诊断的任务，格式为 "源域:目标域"；一难一易，可按需增删
TASKS=(
    "art:clipart"
    "product:realworld"
)

# 训练输出目录（快照存在 runs/process/Diag/<任务>/ 下）和诊断结果目录
TRAIN_ROOT="DiagFix"     # 类名修复后的快照；旧的 runs/process/Diag/ 是修复前的，不能再用
RESULT_DIR="runs/diag_results/$(date +%Y_%m_%d)"

# =============================================================================
# 以下一般不需要修改
# =============================================================================

# 切换到脚本所在目录（即项目根目录）
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）
mkdir -p "$RESULT_DIR"

abbr() {
    case "$1" in
        art) echo "Ar" ;; clipart) echo "Cl" ;; product) echo "Pr" ;; realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

for pair in "${TASKS[@]}"; do
    s="${pair%%:*}"
    t="${pair##*:}"
    task="$(abbr "$s")2$(abbr "$t")"
    train_dir="runs/process/$TRAIN_ROOT/$task"

    # ---------------------------------------------------------------------
    # 第 1 步：训练并保存快照
    # ---------------------------------------------------------------------
    last="${STAGES[${#STAGES[@]}-1]}"
    if [ -f "$train_dir/snapshot_${last}.pt" ]; then
        echo "==> ${task}：快照已存在，跳过训练"
    else
        # main.py 发现同名目录会自动加时间戳另建，导致这里找不到快照，所以提前拦下
        if [ -d "$train_dir" ]; then
            echo "[错误] ${train_dir} 已存在但没有完整快照（可能上次中断），请删除后重跑"; exit 1
        fi
        echo "==> ${task}：训练到 ${STOP_AT}，保存快照 ${SNAPSHOT_AT}"
        # $SNAPSHOT_AT 不加引号，让它拆成多个参数
        $PYTHON main.py \
            --device "$GPU" \
            --name "$TRAIN_ROOT/$task" \
            --epochs "$EPOCHS" \
            --dataset "$DATASET" \
            --source "$s" \
            --target "$t" \
            --snapshot_at $SNAPSHOT_AT \
            --stop_at "$STOP_AT"
    fi

    # ---------------------------------------------------------------------
    # 第 2 步：逐个快照诊断
    # ---------------------------------------------------------------------
    for stage in "${STAGES[@]}"; do
        out="$RESULT_DIR/$task/$stage"
        if [ -f "$out/summary.json" ]; then
            echo "==> ${task} ${stage}：已有结果，跳过"
            continue
        fi
        echo "==> ${task} ${stage}：诊断"
        mkdir -p "$out"
        $PYTHON tools/diag_ot.py \
            --snapshot "$train_dir/snapshot_${stage}.pt" \
            --out "$out" \
            --device "$GPU" \
            --batch_size "$DIAG_BATCH" \
            --seeds $DIAG_SEEDS \
            --scales $DIAG_SCALES \
            --regs $DIAG_REGS \
            2>&1 | tee "$out/diag_log.txt"
    done
done

# -----------------------------------------------------------------------------
# 第 3 步：汇总并打包
# -----------------------------------------------------------------------------
echo "==> 汇总"
$PYTHON tools/diag_collect.py --results "$RESULT_DIR" --process "runs/process/$TRAIN_ROOT"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
