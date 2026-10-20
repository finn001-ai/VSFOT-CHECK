#!/usr/bin/env bash
# =============================================================================
# 第 1 步：老师文本端加可学习的提示上下文（2026-10-20），只在 Ar→Pr 上（CLIP 强、准确率 90 上下的代表任务）
#   DIFO / DIFO++ / ProDe / DUet / COSMO 都在目标域上学习文本提示，我们的老师文本端一直是固定的 "a photo of a {类名}"；
#   H1：学生「探针 − 实际」与老师几乎一一对应（Ar→Pr 3.6 / 3.9），缺口来自老师固定的文本原型分类头。
#   上下文 = 4 个所有类共享的向量，用 "a photo of a" 的词嵌入初始化（初始输出与固定提示完全相同），学习目标沿用 MGVA（学生 top-3）：
#     cap02_prompt      提示上下文 + 图像侧 adapter，上下文学习率 1e-5（Adam，adapter 为 5e-5）
#     cap02_prompt5     同上，上下文学习率 5e-5
#     cap02_promptonly  只学提示上下文，adapter 不更新（同 COSMO），学习率 1e-5
#     both_cap02_dyn    对照：cap02 加训练动态记录（不影响训练，最后 5 次应等于 91.91），用来比较老师准确率的变化
#   4 组都开 --log_dynamics（每个 epoch 记录老师准确率、一致率等，每组多约 8 分钟），共约 2.2 小时。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_prompt.sh
# 中途想先看已完成的部分：!SUMMARY_ONLY=1 bash scripts/run_prompt.sh
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

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"；按重要性排序，不想跑的组在前面加 # 注释掉
GRPS=(
    "cap02_prompt|--t_prompt --t_prompt_lr 1e-5 --log_dynamics"
    "cap02_prompt5|--t_prompt --t_prompt_lr 5e-5 --log_dynamics"
    "cap02_promptonly|--t_prompt --no_adapter --t_prompt_lr 1e-5 --log_dynamics"
    "both_cap02_dyn|--log_dynamics"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_prompt"
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
modes = ["cap02_prompt", "cap02_prompt5", "cap02_promptonly", "both_cap02_dyn", "cap02_proto", "cap02_tstrong"]
print("## 第 1 步：可学习提示上下文（Ar→Pr；both_cap02_dyn 应为 +0.00，后两行为之前的参考）：相对 cap02 的差（seed 42，最后 5 次评估均值，百分点；括号内为用时）\n")
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
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |cap02_prompt|both_cap02_dyn|COSMO" || true
echo ""
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## Ar2Pr$/{t=$2} t!="" && /^- (both_cap02|cap02_prompt[a-z0-9]*|both_cap02_dyn):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo ""
echo "训练动态（Ar→Pr；每行一个评估点）："
awk '/^## 训练动态：/{p = ($0 ~ /：(both_cap02_dyn|cap02_prompt[a-z0-9]*) \/ Ar2Pr_s42$/)} p' "$RESULT_DIR/collect.txt" || true
echo "============================== 汇总结束 =============================="
