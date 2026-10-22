#!/usr/bin/env bash
# =============================================================================
# cap02 + tstrong 补齐 Office-Home 12 个任务（2026-10-22），目标：12 任务平均超过 COSMO 86.41
#   Ar→Pr 上二十多组改动都在 92 ± 0.3（40 epoch 确认 fm80_mix −0.14、aad −0.31），混淆类拆分的真实标签上限也只到约 93.5，
#   改为看 12 任务平均：cap02（both_cap02）86.23，COSMO 86.41。tstrong（老师在强增强视图上也学同一个目标）在
#   Ar→Cl / Ar→Pr / Cl→Ar / Re→Ar 上相对 cap02 平均 +0.41（archive/2026_10_20_teacher），这里补剩下 8 个任务。
#   设置与 run_teacher.sh 的 cap02_tstrong 相同（40 epoch，seed 42），组名相同，汇总时自动合并已有的 4 个任务。
#
#   !bash scripts/run_tstrong12.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_tstrong12.sh
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
GRPS=(
    "cap02_tstrong|--t_strong"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:clipart art:product art:realworld clipart:art clipart:product clipart:realworld \
                product:art product:clipart product:realworld realworld:art realworld:clipart realworld:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_tstrong12"
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
$PYTHON - "$RESULT_DIR/runs.csv" <<'EOF' || echo "[警告] 差值表生成失败"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["seed"] == "42" and r["last5"]]
acc = {(r["mode"], r["task"]): float(r["last5"]) * 100 for r in rows}
cosmo = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Ar2Re": 92.0, "Cl2Ar": 85.7, "Cl2Pr": 92.4, "Cl2Re": 91.9,
         "Pr2Ar": 84.3, "Pr2Cl": 76.1, "Pr2Re": 91.7, "Re2Ar": 85.6, "Re2Cl": 75.5, "Re2Pr": 92.7}
tasks = list(cosmo)
f = lambda v: "-" if v is None else f"{v:.2f}"
print("## cap02 + tstrong：Office-Home 12 个任务（40 epoch，seed 42，最后 5 次评估均值）\n")
print("| 任务 | cap02 | cap02_tstrong | 差 | COSMO | tstrong − COSMO |")
print("|---|---|---|---|---|---|")
for t in tasks:
    a, b = acc.get(("both_cap02", t)), acc.get(("cap02_tstrong", t))
    d1 = "-" if None in (a, b) else f"{b - a:+.2f}"
    d2 = "-" if b is None else f"{b - cosmo[t]:+.2f}"
    print(f"| {t} | {f(a)} | {f(b)} | {d1} | {cosmo[t]} | {d2} |")
done = [t for t in tasks if ("cap02_tstrong", t) in acc and ("both_cap02", t) in acc]
if done:
    m = lambda k: sum(acc[(k, t)] for t in done) / len(done)
    c = sum(cosmo[t] for t in done) / len(done)
    print(f"\n已完成 {len(done)} / 12 个任务；在这些任务上平均：cap02 {m('both_cap02'):.2f}，cap02_tstrong {m('cap02_tstrong'):.2f}，COSMO {c:.2f}")
EOF
echo "============================== 汇总结束 =============================="
