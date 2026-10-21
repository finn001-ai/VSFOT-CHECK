#!/usr/bin/env bash
# =============================================================================
# 方向 B：COSMO 式共享共识（2026-10-22），只在 Ar→Pr 上
#   方向 A（run_gate.sh）：MGVA 门控 agree / conf5 / conf8 = −0.14 / −0.10 / +0.13（20 epoch），无效。
#   所有变体里「任一判对」在第 138 步到 93.7 ~ 94.1，之后掉回 92.5 ~ 92.8，学生停在约 92；第 1 步：学生终点 ≈ 训练前并集 + 0.5。
#   改「谁教谁」：老师和学生都学同一个共识 = 两边（更新前）预测按熵加权的专家乘积（COSMO 消融：共享共识比互相监督 +1.0）。
#   底座同 run_gate.sh 的 base_sgd（cap02 + adapter + 标准提示上下文 SGD），对照 base_sgd_e20：
#     cons_b        --consensus both：老师学共识（代替学生 top-3），学生的对齐目标也换成共识
#     cons_b_po     同上，老师只学提示上下文（adapter 冻结；第 1 步只学提示的老师保留互补性，并集 93.15）
#     cons_b_po_ce  同上，学生再加对共识 argmax 的硬交叉熵（权重 0.4，COSMO 的设置；COSMO 去掉硬 CE 掉 6.7）
#     cons_t        --consensus teacher：只换老师的目标，学生不变
#   默认 20 epoch 快速筛选（组名带 _e20），每组约 26 分钟；FULL=1 跑完整 40 epoch。
#
#   !bash scripts/run_cons.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_cons.sh
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
    "cons_b$SFX|$PSTD --consensus both"
    "cons_b_po$SFX|$PSTD --no_adapter --consensus both"
    "cons_b_po_ce$SFX|$PSTD --no_adapter --consensus both --s_ce 0.4"
    "cons_t$SFX|$PSTD --consensus teacher"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_cons"
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
print(f"## 方向 B：共享共识（Ar→Pr，{'20' if sfx else '40'} epoch，seed 42，最后 5 次评估均值）\n")
print("| 组 | Ar2Pr | 相对 " + base + " | 用时 |")
print("|---|---|---|---|")
for m in [base] + [g + sfx for g in ("cons_b", "cons_b_po", "cons_b_po_ce", "cons_t")]:
    if (m, t) in acc:
        d = f"{acc[(m, t)] - acc[(base, t)]:+.2f}" if (base, t) in acc and m != base else "-"
        print(f"| {m} | {acc[(m, t)]:.2f} | {d} | {sec[(m, t)] / 60:.0f} 分 |")
    else:
        print(f"| {m} | - | - | - |")
print("\n参考（40 epoch）：" + "，".join(f"{m} {acc[(m, t)]:.2f}" for m in ["both_cap02", "both_cap02_dyn", "cap02_prompt5", "base_sgd"] if (m, t) in acc) + "，COSMO 92.6")
EOF
echo ""
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |both_cap02_dyn|base_sgd|cons_|COSMO" || true
echo ""
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## Ar2Pr$/{t=$2} t!="" && /^- (both_cap02|both_cap02_dyn|base_sgd[a-z0-9_]*|cons_[a-z0-9_]*):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo ""
echo "训练动态（Ar→Pr；每行一个评估点）："
awk '/^## 训练动态：/{p = ($0 ~ /：(both_cap02_dyn|base_sgd[a-z0-9_]*|cons_[a-z0-9_]*) \/ Ar2Pr_s42$/)} p' "$RESULT_DIR/collect.txt" || true
echo "============================== 汇总结束 =============================="
