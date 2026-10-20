#!/usr/bin/env bash
# =============================================================================
# 两个改动，只在 Ar→Pr 上（2026-10-20）：Ar→Pr 代表「CLIP 强、准确率 90 上下」的一类任务（同 VisDA-C / DomainNet-126 这类设定）。
#   都以 cap02 为底座（tstrong 在 Ar→Pr 上 −0.29，不叠加，便于归因）：
#     cap02_prior  类均衡的目标从均匀分布改成 CLIP 零样本在目标域上的平均概率 μ̂（--clsw_mode prior，权重 = μ̂ / 学生预测频率，
#                  每步把损失量级调成与原方法相同）。sqrt 在 Art +1.2 / Pr −1.5：固定的均衡力度不对，改由数据决定推向哪里
#     cap02_proto  老师的类原型变成可学习参数（--t_proto，文本嵌入初始化，和 adapter 一起由 MGVA 更新，--t_proto_reg 1 拉向初始文本嵌入）。
#                  H1：学生「探针 − 实际」与老师几乎一一对应（Ar→Pr 3.6 / 3.9），学生继承的是固定文本原型分类头的错误
#   开头先检查 μ̂ 的前提：μ̂ 是否比均匀分布更接近真实类分布（读取 run_h1.sh 的训练前快照，只用标签做检查，几秒）
#   2 组训练约 50 分钟。对照 = 已有的 both_cap02（91.91），参考 cap02_tstrong / cap02_clswsqrt。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_ab.sh
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
    "cap02_prior|--clsw_mode prior"
    "cap02_proto|--t_proto --t_proto_reg 1"
)
JOBS=()
for g in "${GRPS[@]}"; do
    for pair in art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录（汇总时自动包含 both_cap02 对照）
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_ab"
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
echo "## μ̂ 前提检查（训练前快照；L1 越小越接近真实类分布，最大 2）"
snaps=()
for t in Ar2Cl Cl2Ar Ar2Pr Ar2Re; do
    f="runs/H1Probe/p000/${t}_s42/snapshot_p000.pt"
    [ -f "$f" ] && snaps+=("$f")
done
[ ${#snaps[@]} -gt 0 ] && { $PYTHON tools/check_prior.py "${snaps[@]}" || echo "[警告] check_prior.py 失败"; } || echo "[跳过] 找不到 run_h1.sh 的训练前快照"
echo ""
# 相对 cap02（seed 42）的差：每格 = 最后 5 次评估均值之差（百分点），括号内为该组用时
$PYTHON - "$RESULT_DIR/runs.csv" <<'EOF' || echo "[警告] 差值表生成失败"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["seed"] == "42" and r["last5"]]
acc = {(r["mode"], r["task"]): float(r["last5"]) * 100 for r in rows}
sec = {(r["mode"], r["task"]): float(r["elapsed_sec"]) if r["elapsed_sec"] not in ("", "None") else 0.0 for r in rows}
tasks = ["Ar2Pr"]
cosmo = {"Ar2Pr": 92.6}
modes = ["cap02_prior", "cap02_proto", "cap02_tstrong", "cap02_clswsqrt"]
print("## Ar→Pr 上的两个改动（后两行为参考）：相对 cap02 的差（seed 42，最后 5 次评估均值，百分点；括号内为用时）\n")
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
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt" | grep -E "^\||^## 跨任务" | grep -E "跨任务|^\| 模式|^\|---|both_cap02 |cap02_prior|cap02_proto|cap02_tstrong|cap02_clswsqrt|COSMO" || true
echo ""
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## Ar2Pr$/{t=$2} t!="" && /^- (both_cap02|cap02_prior|cap02_proto|cap02_tstrong|cap02_clswsqrt):/{print t": "$0}' "$RESULT_DIR/collect.txt" || true
echo "============================== 汇总结束 =============================="
