#!/usr/bin/env bash
# =============================================================================
# 混淆类拆分纠正：40 epoch 确认（2026-10-22），Ar→Pr
#   20 epoch 上 sfix_tr_rel30（进度 0.1 ~ 0.25 每 epoch 重做改名，0.3 撤掉）92.50，对照 91.91；进度 0.3 之后改名只会改坏。
#   40 epoch 按迭代步对齐（iter 138 ~ 345 每 69 步重做，414 撤掉）= 进度 0.05 ~ 0.125，0.15 撤掉：
#     cap02_sfix          底座 cap02（对照 both_cap02 91.91）
#     cap02_tstrong_sfix  底座 cap02 + tstrong（对照 cap02_tstrong 91.63），即 12 任务计划的配置
#
#   !bash scripts/run_sfix40.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_sfix40.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40; SFX=""   # 40 = README 命令的设置

BASE="--align_mode ot --knn_k 10 --align_view both"

CAP="--align_weight cap --align_cap 0.2"

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"；按重要性排序，不想跑的组在前面加 # 注释掉
SFIX="--split_at 0.05 --split_until 0.125 --split_teacher --split_release 0.15"
GRPS=(
    "cap02_sfix|$SFIX"
    "cap02_tstrong_sfix|--t_strong $SFIX"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_sfix40"
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
t = "Ar2Pr"
print("## 混淆类拆分纠正（Ar→Pr，40 epoch，seed 42，最后 5 次评估均值）\n")
print("| 组 | Ar2Pr | 对照 | 相对 | 用时 |")
print("|---|---|---|---|---|")
for m, base in [("cap02_sfix", "both_cap02"), ("cap02_tstrong_sfix", "cap02_tstrong")]:
    if (m, t) in acc and (base, t) in acc:
        print(f"| {m} | {acc[(m, t)]:.2f} | {base} {acc[(base, t)]:.2f} | {acc[(m, t)] - acc[(base, t)]:+.2f} | {sec[(m, t)] / 60:.0f} 分 |")
    else:
        print(f"| {m} | - | {base} | - | - |")
print("\n参考：20 epoch sfix_tr_rel30 92.50（对照 base_sgd_e20 91.91），COSMO 92.6")
EOF
echo ""
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |cap02_tstrong|sfix|COSMO" || true
echo ""
echo "训练曲线（对照与本轮的组）："
awk '/^## /{t=""} /^## Ar2Pr$/{t=$2} t!="" && /^- (both_cap02|cap02_tstrong|cap02_sfix|cap02_tstrong_sfix):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo ""
echo "训练动态（Ar→Pr；每行一个评估点）："
awk '/^## 训练动态：/{p = ($0 ~ /：(cap02_sfix|cap02_tstrong_sfix) \/ Ar2Pr_s42$/)} p' "$RESULT_DIR/collect.txt" || true
echo ""
echo "改名记录："
grep -H "\[split\]" runs/$TRAIN_ROOT/cap02*sfix/Ar2Pr_s42/run_log.txt 2>/dev/null | sed 's#runs/[^/]*/##; s#/Ar2Pr_s42/run_log.txt:# #' || true
echo "============================== 汇总结束 =============================="
