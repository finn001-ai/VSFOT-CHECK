#!/usr/bin/env bash
# =============================================================================
# 混淆类拆分纠正接进训练（2026-10-22），Ar→Pr
#   tools/diag_split.py：p100 快照上无标签改名全部失败（概率饱和），早期快照为正——40 epoch 调度进度 5%（iter 138，学生 84.73）
#   在老师 CLIP 特征空间类内拆 3 个子簇、第二名比例 ≥ 0.2 且子簇中心离第二名类中心更近就改名：+2.79（纠正 160 / 改坏 36，上限 +3.27），
#   进度 10% +1.40，25% 起失效。这里在训练中做：20 epoch 的进度 0.1 = iter 138，改名样本在之后的训练里对齐计划换成改名后的类。
#     sfix       --split_at 0.1（只改学生的对齐目标）
#     sfix_t     同上，老师（MGVA）也学改名后的类
#     sfix_tr    同上，并在之后每个 epoch 重做一次直到进度 0.25
#   底座同 base_sgd，对照 base_sgd_e20 91.91。run_log.txt 里 [split] 行记录每次改名的纠正 / 改坏数。
#
#   !bash scripts/run_splitfix.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_splitfix.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
FULL="${FULL:-0}"
if [ "$FULL" = "1" ]; then EPOCHS=40; SFX=""; else EPOCHS=20; SFX="_e20"; fi   # 40 = README 命令的设置

BASE="--align_mode ot --knn_k 10 --align_view both"

CAP="--align_weight cap --align_cap 0.2"
PSTD="--t_prompt --t_prompt_opt sgd --log_dynamics"     # 标准提示上下文（DIFO 系列的优化器设置）

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"；按重要性排序，不想跑的组在前面加 # 注释掉
GRPS=(
    "base_sgd$SFX|$PSTD"
    "sfix_t$SFX|$PSTD --split_at 0.1 --split_teacher"
    "sfix$SFX|$PSTD --split_at 0.1"
    "sfix_tr$SFX|$PSTD --split_at 0.1 --split_until 0.25 --split_teacher"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_splitfix"
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
# 相对同一训练长度的 base_sgd（seed 42）的差：每格 = 最后 5 次评估均值之差（百分点）
$PYTHON - "$RESULT_DIR/runs.csv" "$SFX" <<'EOF' || echo "[警告] 差值表生成失败"
import csv, sys
sfx = sys.argv[2]
rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["seed"] == "42" and r["last5"]]
acc = {(r["mode"], r["task"]): float(r["last5"]) * 100 for r in rows}
sec = {(r["mode"], r["task"]): float(r["elapsed_sec"]) if r["elapsed_sec"] not in ("", "None") else 0.0 for r in rows}
t, base = "Ar2Pr", "base_sgd" + sfx
print(f"## 混淆类拆分纠正（Ar→Pr，{'20' if sfx else '40'} epoch，seed 42，最后 5 次评估均值）\n")
print("| 组 | Ar2Pr | 相对 " + base + " | 用时 |")
print("|---|---|---|---|")
for m in [base] + [g + sfx for g in ("sfix_t", "sfix", "sfix_tr")]:
    if (m, t) in acc:
        d = f"{acc[(m, t)] - acc[(base, t)]:+.2f}" if (base, t) in acc and m != base else "-"
        print(f"| {m} | {acc[(m, t)]:.2f} | {d} | {sec[(m, t)] / 60:.0f} 分 |")
    else:
        print(f"| {m} | - | - | - |")
print("\n参考（40 epoch）：" + "，".join(f"{m} {acc[(m, t)]:.2f}" for m in ["both_cap02", "both_cap02_dyn", "cap02_prompt5", "base_sgd"] if (m, t) in acc) + "，COSMO 92.6")
EOF
echo ""
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |both_cap02_dyn|base_sgd|sfix|COSMO" || true
echo ""
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## Ar2Pr$/{t=$2} t!="" && /^- (both_cap02|both_cap02_dyn|base_sgd[a-z0-9_]*|sfix[a-z0-9_]*):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo ""
echo "训练动态（Ar→Pr；每行一个评估点）："
awk '/^## 训练动态：/{p = ($0 ~ /：(both_cap02_dyn|base_sgd[a-z0-9_]*|sfix[a-z0-9_]*) \/ Ar2Pr_s42$/)} p' "$RESULT_DIR/collect.txt" || true
echo ""
echo "改名记录："
grep -H "\[split\]" runs/$TRAIN_ROOT/sfix*$SFX/Ar2Pr_s42/run_log.txt 2>/dev/null | sed 's#runs/[^/]*/##; s#/Ar2Pr_s42/run_log.txt:# #' || true
echo "============================== 汇总结束 =============================="
