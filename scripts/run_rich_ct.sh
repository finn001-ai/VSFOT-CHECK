#!/usr/bin/env bash
# =============================================================================
# 2026-10-28 塌缩过程记录：cap02_tstrong + 老师侧共识 + --log_dynamics --log_samples --log_rich，40 epoch，seed 42，串行
#   加共识后 →Clipart 整类消失（Ar2Cl：Computer 预测数 107 → 0，被 Monitor 吞；Pencil / Marker → Pen），Re2Ar 方向相反（Monitor → Computer，净收益）。
#   记完整概率，看类是先在老师还是学生上消失、何时开始、之前有没有可提前看到的迹象。最后 5 次 OA 应与 75.88 / 85.23 一致（校验）。
#
#   !bash scripts/run_rich_ct.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
GROUP="GateRich/cap02_tstrong_ct"
EXTRA="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --log_dynamics --log_samples --log_rich --consensus teacher"

PAIRS="art:clipart realworld:art"
abbr() { case "$1" in art) echo Ar ;; clipart) echo Cl ;; product) echo Pr ;; realworld) echo Re ;; esac; }
mkdir -p runs/rich_ct_logs

T0=$(date +%s)
for pair in $PAIRS; do
    s="${pair%%:*}"; t="${pair##*:}"; task="$(abbr "$s")2$(abbr "$t")"
    dir="$GROUP/${task}_s42"
    if [ -f "runs/$dir/.done" ]; then echo "==> $task：已完成，跳过"; continue; fi
    if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
    echo "==> 开始 $task（$(date +%H:%M)）"
    # $EXTRA 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个任务
    if $PYTHON main.py --device "$GPU" --name "$dir" --dataset OfficeHomeDataset --source "$s" --target "$t" --seed 42 $EXTRA \
        > "runs/rich_ct_logs/${task}.log" 2>&1; then
        touch "runs/$dir/.done"; echo "==> 完成 $task"
    else
        echo "[错误] $task 训练失败，见 runs/rich_ct_logs/${task}.log"
    fi
done
echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"

echo "============================== 汇总开始 =============================="
$PYTHON - <<'PY'
import os, re
ref = {"Ar2Cl": 75.88, "Re2Ar": 85.23}
for t, r in ref.items():
    p = f"runs/GateRich/cap02_tstrong_ct/{t}_s42/Detail.txt"
    if not os.path.exists(p):
        print(t, "-"); continue
    v = [float(x) * 100 for x in re.findall(r"Overall Accuracy \(OA\): ([0-9.]+)", open(p).read())][1:]
    print(f"{t}: 后 5 {sum(v[-5:]) / len(v[-5:]):.2f}（cap02_tstrong_ct {r}），rich.npz {os.path.exists(os.path.dirname(p) + '/rich.npz')}")
PY
echo "============================== 汇总结束 =============================="
