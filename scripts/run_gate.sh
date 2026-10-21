#!/usr/bin/env bash
# =============================================================================
# 方向 A：MGVA 只在学生可靠的样本上教 adapter（2026-10-21），只在 Ar→Pr 上
#   第 1 步（archive/2026_10_21_prompt）：adapter 学每个样本锐化后的学生 top-3，第 69 步老师 87.1 → 84.5（熵 0.58 → 1.33），
#   结束时一致率 99%、并集只比学生高 0.3 —— adapter 记住的是学生（包括错误），不是目标域；只学提示的老师用同样的目标不掉、不塌缩。
#   底座：cap02 + adapter + 标准提示上下文（DIFO / ProDe / DIFO++ 公开代码的设置：CoOp 4 个上下文向量、"a photo of a" 初始化，
#   SGD 动量 0.9、wd 5e-4、lr = 主学习率 × 0.1 = 1e-3）。只改 MGVA 用哪些样本，损失在保留样本上取平均（量级不变）：
#     base_sgd    底座（MGVA 用全部样本），本轮的对照
#     gate_agree  学生 top-1 和冻结的零样本 CLIP 一致的样本
#     gate_conf5  学生置信度在其预测类内排前 50%（按学生记忆库逐类取分位数）
#     gate_conf8  同上，前 80%
#   都开 --log_dynamics，训练动态新增「MGVA 保留」比例和「保留样本学生准确率」（标签只用于记录）。每组约 36 分钟，共约 2.4 小时。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_gate.sh
# 中途想先看已完成的部分：!SUMMARY_ONLY=1 bash scripts/run_gate.sh
# 把「汇总开始」到「汇总结束」之间贴回即可，不用下载
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40              # README 命令的设置

BASE="--align_mode ot --knn_k 10 --align_view both"

CAP="--align_weight cap --align_cap 0.2"
PSTD="--t_prompt --t_prompt_opt sgd --log_dynamics"     # 标准提示上下文（DIFO 系列的优化器设置）

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"；按重要性排序，不想跑的组在前面加 # 注释掉
GRPS=(
    "base_sgd|$PSTD"
    "gate_agree|$PSTD --mgva_gate agree"
    "gate_conf5|$PSTD --mgva_gate conf --mgva_keep 0.5"
    "gate_conf8|$PSTD --mgva_gate conf --mgva_keep 0.8"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_gate"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

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

# 先展开成「源域 目标域 任务名 组名 种子 | 额外参数」，并统计还需要训练几组
todo=()
n_done=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r name pair seed extra <<< "$job"
    s="${pair%%:*}"
    t="${pair##*:}"
    task="$(abbr "$s")2$(abbr "$t")"
    if [ -f "runs/$TRAIN_ROOT/$name/${task}_s${seed}/.done" ]; then
        echo "==> ${task} ${name} seed=${seed}：已完成，跳过"
        n_done=$((n_done + 1))
    else
        todo+=("$s $t $task $name $seed|$extra")
    fi
done
echo "==> 已完成 ${n_done} 组，本次需要训练 ${#todo[@]} 组"
[ "$SUMMARY_ONLY" = "1" ] && { echo "==> SUMMARY_ONLY=1：跳过训练"; todo=(); }

k=0
# ${todo[@]+...}：数组为空时不展开，避免 set -u 在旧版 bash 下报错
for item in ${todo[@]+"${todo[@]}"}; do
    read -r s t task name seed <<< "${item%%|*}"
    extra="${item#*|}"
    k=$((k + 1))
    dir="runs/$TRAIN_ROOT/$name/${task}_s${seed}"

    # main.py 遇到同名目录会自动加时间戳另建，导致汇总时找不到，所以提前拦下
    if [ -d "$dir" ]; then
        echo "[错误] ${dir} 已存在但没有完成标记（可能上次中断），请删除后重跑"; exit 1
    fi

    echo ""
    echo "################################################################"
    echo "==> 第 ${k}/${#todo[@]} 组：${task}（${s} -> ${t}），${name}，seed=${seed}"
    echo "    参数：${extra}"
    echo "################################################################"
    # $extra 不加引号，让它拆成多个参数
    $PYTHON main.py \
        --device "$GPU" \
        --name "$TRAIN_ROOT/$name/${task}_s${seed}" \
        --epochs "$EPOCHS" \
        --dataset "$DATASET" \
        --source "$s" \
        --target "$t" \
        --seed "$seed" \
        $extra
    # 训练正常结束后才写完成标记
    touch "$dir/.done"
done

echo "==> 汇总"
mkdir -p "$RESULT_DIR"
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR" | grep -v "^已打包" > "$RESULT_DIR/collect.txt" \
    || echo "[警告] ablation_collect.py 失败"
echo "============================== 汇总开始（把这一段完整贴回）=============================="
# 相对 cap02（seed 42）的差：每格 = 最后 5 次评估均值之差（百分点），括号内为该组用时
$PYTHON - "$RESULT_DIR/runs.csv" <<'EOF' || echo "[警告] 差值表生成失败"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["seed"] == "42" and r["last5"]]
acc = {(r["mode"], r["task"]): float(r["last5"]) * 100 for r in rows}
sec = {(r["mode"], r["task"]): float(r["elapsed_sec"]) if r["elapsed_sec"] not in ("", "None") else 0.0 for r in rows}
tasks = ["Ar2Pr"]
cosmo = {"Ar2Pr": 92.6}
modes = ["base_sgd", "gate_agree", "gate_conf5", "gate_conf8", "both_cap02_dyn", "cap02_prompt5", "cap02_promptonly"]
print("## 方向 A：MGVA 只在可靠样本上教 adapter（Ar→Pr；后三行为第 1 步的参考）：相对 cap02 的差（seed 42，最后 5 次评估均值，百分点；括号内为用时）\n")
print("| 组 | " + " | ".join(tasks) + " | 平均差 | 任务数 |")
print("|---|" + "---|" * (len(tasks) + 2))
print("| cap02（对照，绝对值） | " + " | ".join(f"{acc[('both_cap02', t)]:.2f}" if ("both_cap02", t) in acc else "-" for t in tasks) + " | - | - |")
print("| COSMO（绝对值） | " + " | ".join(f"{cosmo[t]:.1f}" for t in tasks) + " | - | - |")
for m in modes:
    cells, ds = [], []
    for t in tasks:
        if (m, t) in acc and ("both_cap02", t) in acc:
            d = acc[(m, t)] - acc[("both_cap02", t)]
            ds.append(d)
            cells.append(f"{d:+.2f}（{sec[(m, t)] / 60:.0f} 分）")
        else:
            cells.append("-")
    print(f"| {m} | " + " | ".join(cells) + f" | {sum(ds) / len(ds):+.2f} | {len(ds)} |" if ds else f"| {m} | " + " | ".join(cells) + " | - | 0 |")
print("\ncap02 对照用时：" + "，".join(f"{t} {sec[('both_cap02', t)] / 60:.0f} 分" for t in tasks if ("both_cap02", t) in sec))
EOF
echo ""
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |both_cap02_dyn|base_sgd|gate_|COSMO" || true
echo ""
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## Ar2Pr$/{t=$2} t!="" && /^- (both_cap02|both_cap02_dyn|base_sgd|gate_[a-z0-9]*):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo ""
echo "训练动态（Ar→Pr；每行一个评估点）："
awk '/^## 训练动态：/{p = ($0 ~ /：(both_cap02_dyn|base_sgd|gate_[a-z0-9]*) \/ Ar2Pr_s42$/)} p' "$RESULT_DIR/collect.txt" || true
echo "============================== 汇总结束 =============================="
