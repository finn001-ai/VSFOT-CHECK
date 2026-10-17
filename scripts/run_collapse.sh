#!/usr/bin/env bash
# =============================================================================
# 共识塌缩实验：训练在第 748 步左右停滞的原因与干预（2026-10-17）
#   训练动态诊断发现：伪标签从第 204 步起就不比学生好；老师（带 adapter 的 CLIP）的进步全部来自学生，
#   两者收敛到同一个不动点（一致率 88% → 97%），并集（任一判对）从峰值 77–79% 掉回 75–77%，
#   约 115 个「只有一方判对」的样本最后变成两边都错。
#   四组都以目前最好的 ot_knn10_both 为底座，都记录训练动态和逐样本预测（samples.npz）：
#     both_samples     不干预（对照，准确率应和 ot_knn10_both 的 75.98 完全一致）；
#                      状态转移回答：丢失的样本是学生吸收了老师的错误，还是老师吸收了学生的错误
#     both_freezeT020  第 544 步（并集最高点附近）后冻结老师：并集不再下降 → 漏洞在老师侧
#     both_adapter12   CLIP 全部 12 个 block 都插 adapter：老师容量变大后，不动点会不会一起抬高
#                      （adapter 初始化多消耗随机数，所以从第 0 步起就和底座不同）
#     both_selfknn025  进度 0.25 后近邻平滑改用 (CLIP + 学生) / 2：对齐目标高于学生之后，能否越过不动点
#   第 2 轮（--align_agree_after）：到该进度后只在学生预测 = 计划 argmax 的样本上做对齐，分歧样本上不再把学生拉向老师
#     both_agree010 / both_agree025   第 272 / 680 步起
#
# 结果写到 runs/AblationFix/<组名>/<任务>_s<种子>/，汇总表末尾附「训练动态」表，另有 collapse.md（状态转移）
#
# 在 Jupyter 里运行（项目根目录下）：
#   !bash scripts/run_collapse.sh
# 跑完下载 runs/ablation_results/<日期>.zip
#
# 已完成的组（有 .done 标记）会自动跳过；耗时参考：每组约 50 分钟（含每个 epoch 一次全量前向）
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASET="OfficeHomeDataset"
PYTHON="python -u"
GPU=0
EPOCHS=40              # README 命令的设置；这一轮要看第 748 步之后曲线还动不动，所以不截断

SEEDS=(42)             # 和 AblationFix 基线相同的种子

BASE="--align_mode ot --knn_k 10 --align_view both --log_dynamics --log_samples"

# 实验组，格式 "组名|额外参数"；不想跑的行在前面加 # 注释掉
RUN_GROUPS=(
    # 第 1 轮（2026-10-17，已完成，会自动跳过）：漏洞在学生侧；冻结老师 / 12 个 adapter / 混入学生概率都无效
    "both_samples|$BASE"
    # "both_freezeT020|$BASE --freeze_teacher_at 0.2"
    # "both_adapter12|$BASE --adaptercount 12"
    # "both_selfknn025|$BASE --knn_self_after 0.25 --knn_self_w 0.5"
    # 第 2 轮：后期只在师生一致的样本上对齐（分歧样本上学生判对的次数约为老师的 2 倍，交叉点在第 204 步左右）
    # "both_agree010|$BASE --align_agree_after 0.1"     # 第 272 步起
    # "both_agree025|$BASE --align_agree_after 0.25"    # 第 680 步起
)

# 先只看 Ar→Cl
TASKS=(
    "art:clipart"
)

# 和 scripts/run_ablation.sh 用同一个目录，汇总时自动包含基线
TRAIN_ROOT="AblationFix"
RESULT_DIR="runs/ablation_results/$(date +%Y_%m_%d)"

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
for seed in "${SEEDS[@]}"; do
    for pair in "${TASKS[@]}"; do
        s="${pair%%:*}"
        t="${pair##*:}"
        task="$(abbr "$s")2$(abbr "$t")"
        for g in "${RUN_GROUPS[@]}"; do
            name="${g%%|*}"
            if [ -f "runs/$TRAIN_ROOT/$name/${task}_s${seed}/.done" ]; then
                echo "==> ${task} ${name} seed=${seed}：已完成，跳过"
                n_done=$((n_done + 1))
            else
                todo+=("$s $t $task $name $seed|${g#*|}")
            fi
        done
    done
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
$PYTHON tools/collapse_analysis.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
$PYTHON tools/ablation_collect.py --process "runs/$TRAIN_ROOT" --out "$RESULT_DIR"
echo "==> 完成，下载 ${RESULT_DIR}.zip 发给我即可"
