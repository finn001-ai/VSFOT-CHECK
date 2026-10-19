#!/usr/bin/env bash
# =============================================================================
# 后期减弱 MGVA（学生 → 老师）（2026-10-19）：Art 目标诊断（scripts/run_art.sh）发现，cap02 之后剩下的塌缩是对称的——
#   并集峰值后，学生吸收老师错误与老师吸收学生错误一样多（Cl→Ar 35 / 32，Re→Ar 30 / 27，Ar→Cl 45 / 43），
#   一致率最后到 98%。cap02 只限制了「老师 → 学生」，这一轮把同样的思路用在另一个方向：
#   到进度 P 后把老师 adapter 的学习率乘以 0.2（老师用 Adam，缩放损失无效，所以缩放学习率；完全冻结在底座上 −0.8，所以不冻结）
#     cap02_vil025   P = 0.25
#     cap02_vil050   P = 0.5
#   在 Cl→Ar、Re→Ar、Ar→Cl、Ar→Pr 上和 cap02 对照，每组 18–26 分钟，共 8 组约 3 小时。
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_vil.sh
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

# 训练任务，格式 "组名|源域:目标域|种子|额外参数"
JOBS=()
for g in "cap02_vil025|--vil_after 0.25 --vil_low 0.2" "cap02_vil050|--vil_after 0.5 --vil_low 0.2"; do
    for pair in clipart:art realworld:art art:clipart art:product; do
        JOBS+=("${g%%|*}|$pair|42|$BASE $CAP ${g#*|}")
    done
done

# 和 scripts/run_weight.sh 同一个目录
TRAIN_ROOT="WeightSched"
RESULT_DIR="runs/weight_results/$(date +%Y_%m_%d)_vil"

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
echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR" | grep -v "^已打包" > "$RESULT_DIR/collect.txt"
sed -n '/^## 跨任务对照/,/^## 分解/p' "$RESULT_DIR/collect.txt"
echo "训练曲线（只列 cap02 与本轮的组）："
awk '/^## /{t=""} /^## (Ar2Cl|Ar2Pr|Cl2Ar|Re2Ar)$/{t=$2} t!="" && /^- (both_cap02|cap02_vil[0-9]+):/{print t": "$0}' "$RESULT_DIR/collect.txt"
echo "============================== 汇总结束 =============================="
