#!/usr/bin/env bash
# =============================================================================
# H1（2026-10-20）：CLIP 强的目标域（Art / Product / Real World）上，限制我们的是不是「学生没学到 CLIP 的表示」
#   我们的方法在 CLIP 弱的 Clipart 目标上领先 COSMO，在 CLIP 强的目标上落后；Cl→Ar 上 cap02 学生特征的探针只有 85.9（≈ COSMO 85.7）。
#   如果原始 CLIP 特征的线性探针明显更高，说明 CLIP 表示里有、但只靠概率蒸馏传不给学生的信息 —— 改进方向是让学生学 CLIP 的特征结构。
#   1. 12 个任务各存一个训练前快照（--snapshot_at 0 --stop_at 0：只做初始化，不训练，每个约 2 分钟）
#   2. cap02 训练结束的快照：Cl→Ar、Re→Ar（以及 Ar→Cl，如果有）复用 run_art.sh / run_cap.sh 已有的；
#      Ar→Pr、Ar→Re 重新训练一次 cap02 并存最后的快照（存快照不影响训练，最后 5 次应和 cap02 的 91.91 / 91.31 相同）
#   3. tools/diag_repr.py 分析每个快照（各层线性探针、近邻标签等），tools/h1_collect.py 汇总
#   耗时约 2.5–3 小时（训练 2 组约 50 分钟 + 分析约 17 个快照）。H2 等这一轮结果出来后再决定。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_h1.sh
# 已完成的快照 / 分析会自动跳过；把「汇总开始」到「汇总结束」之间贴回即可，不用下载
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

BASE="--align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2"
PAIRS=(art:clipart product:clipart realworld:clipart clipart:art product:art realworld:art
       art:product clipart:product realworld:product art:realworld clipart:realworld product:realworld)
# 需要重新训练 cap02 以得到训练结束快照的任务
TRAIN_P100=(art:product art:realworld)
# 已有的 cap02 训练结束快照（不存在的会跳过）
EXIST_P100=(Cl2Ar Re2Ar Ar2Cl)

TRAIN_ROOT="H1Probe"
RESULT_DIR="runs/h1_results/h1"

# =============================================================================
# 以下一般不需要修改
# =============================================================================
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

abbr() {
    case "$1" in
        art) echo "Ar" ;; clipart) echo "Cl" ;; product) echo "Pr" ;; realworld) echo "Re" ;;
        *) echo "未知域名: $1" >&2; exit 1 ;;
    esac
}

# 训练（或只做初始化）一组；$1 源域 $2 目标域 $3 组名 $4 额外参数
train() {
    local s="$1" t="$2" name="$3" extra="$4"
    local task="$(abbr "$s")2$(abbr "$t")"
    local dir="runs/$TRAIN_ROOT/$name/${task}_s${SEED}"
    if [ -f "$dir/.done" ]; then
        echo "==> ${task} ${name}：已完成，跳过"; return
    fi
    # main.py 遇到同名目录会自动加时间戳另建，导致找不到快照，所以提前拦下
    if [ -d "$dir" ]; then
        echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
    fi
    echo "==> ${task} ${name}  参数：${extra}"
    $PYTHON main.py --device "$GPU" --name "$TRAIN_ROOT/$name/${task}_s${SEED}" --epochs "$EPOCHS" \
        --dataset "$DATASET" --source "$s" --target "$t" --seed "$SEED" $extra
    touch "$dir/.done"
}

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
    $PYTHON tools/diag_repr.py --snapshot "$snap" --out "$out" --device "$GPU" > "$out/diag_log.txt" 2>&1 \
        || echo "[警告] ${task} ${stage} 分析失败，见 ${out}/diag_log.txt"
}

# 1. 训练前快照
for pair in "${PAIRS[@]}"; do
    train "${pair%%:*}" "${pair##*:}" p000 "$BASE --snapshot_at 0 --stop_at 0"
done
# 2. cap02 训练结束快照
for pair in "${TRAIN_P100[@]}"; do
    train "${pair%%:*}" "${pair##*:}" cap02_snap "$BASE --snapshot_at 1.0"
done

# 3. 分析
for pair in "${PAIRS[@]}"; do
    task="$(abbr "${pair%%:*}")2$(abbr "${pair##*:}")"
    analyse "$task" p000 "runs/$TRAIN_ROOT/p000/${task}_s${SEED}/snapshot_p000.pt"
done
for pair in "${TRAIN_P100[@]}"; do
    task="$(abbr "${pair%%:*}")2$(abbr "${pair##*:}")"
    analyse "$task" p100 "runs/$TRAIN_ROOT/cap02_snap/${task}_s${SEED}/snapshot_p100.pt"
done
for task in "${EXIST_P100[@]}"; do
    analyse "$task" p100 "runs/WeightSched/both_cap02_dyn/${task}_s${SEED}/snapshot_p100.pt"
done

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON tools/h1_collect.py --root "$RESULT_DIR" || echo "[警告] h1_collect.py 失败"
echo "重新训练的 cap02（复现检查，应和 Ar2Pr 91.91 / Ar2Re 91.31 相同）："
for pair in "${TRAIN_P100[@]}"; do
    task="$(abbr "${pair%%:*}")2$(abbr "${pair##*:}")"
    log="runs/$TRAIN_ROOT/cap02_snap/${task}_s${SEED}/run_log.txt"
    [ -f "$log" ] && $PYTHON - "$log" "$task" <<'EOF' || true
import re, sys
acc = [float(m.group(1)) for l in open(sys.argv[1], encoding="utf-8") for m in [re.search(r"Now Acc: ([0-9.]+)", l)] if m]
print(f"- {sys.argv[2]}：最后 5 次均值 {sum(acc[-5:]) / len(acc[-5:]) * 100:.2f}（{len(acc)} 次评估）" if acc else f"- {sys.argv[2]}：日志里没有评估记录")
EOF
done
echo "============================== 汇总结束 =============================="
