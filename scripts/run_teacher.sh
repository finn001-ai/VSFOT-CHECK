#!/usr/bin/env bash
# =============================================================================
# 老师侧的学习信号（2026-10-20）：组件审计（scripts/run_audit.sh）中学生侧组件都在 ±0.6 内，老师不更新 −4.15（4 个任务一致）——
#   最终准确率由老师能学多好决定，而老师（CLIP adapter）唯一的学习信号是学生当前这一批的 top-3。在 cap02 上：
#     cap02_tema     老师学的学生概率换成逐样本跨 epoch 的动量平均（动量 0.7，初始 = 源模型概率）：少吸收学生的临时错误
#     cap02_tstrong  老师在强增强视图上也学同一个目标（与原视图的损失取平均）：视图一致性约束，不只是照抄学生
#     cap02_tim      老师损失加上它自己预测的 IM（熵 + diversity，权重 1）：一个不依赖学生的学习信号
#   另外两组重做类频率重加权的消融（审计里的 noclsw 把对齐损失缩小了约 65 倍，崩溃来自量级而不是去掉均衡）：
#     cap02_clswconst  所有类同一权重（不做均衡），每步把损失量级调成与原方法相同
#     cap02_clswsqrt   1/√频率（均衡推得轻一半），同样量级匹配
#   任务同组件审计：Ar→Cl、Ar→Pr、Cl→Ar、Re→Ar，seed 42，对照 = 已有的 both_cap02。
#   每组 4 个任务约 85 分钟（tstrong 多一次 ViT 前向和反向，会更慢），5 组共约 7.5 小时，按重要性排序。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_teacher.sh
# 中途 GPU 用完 / 想先看已完成的部分：
#   !SUMMARY_ONLY=1 bash scripts/run_teacher.sh      （只汇总，不训练）
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
    "cap02_tema|--t_ema --t_ema_m 0.7"
    "cap02_tstrong|--t_strong"
    "cap02_tim|--t_im 1"
    "cap02_clswconst|--clsw_mode const"
    "cap02_clswsqrt|--clsw_mode sqrt"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:clipart clipart:art realworld:art art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_teacher"
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
tasks = ["Ar2Cl", "Ar2Pr", "Cl2Ar", "Re2Ar"]
cosmo = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Cl2Ar": 85.7, "Re2Ar": 85.6}
modes = ["cap02_tema", "cap02_tstrong", "cap02_tim", "cap02_clswconst", "cap02_clswsqrt", "cap02_nomgva", "cap02_noclsw"]
print("## 老师侧学习信号 + 类频率重加权：相对 cap02 的差（seed 42，最后 5 次评估均值，百分点；括号内为用时）\n")
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
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |cap02_t|cap02_nomgva|cap02_c|COSMO" || true
echo ""
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## (Ar2Cl|Ar2Pr|Cl2Ar|Re2Ar)$/{t=$2} t!="" && /^- (both_cap02|cap02_t[a-z]+|cap02_clsw[a-z]+):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo "============================== 汇总结束 =============================="
