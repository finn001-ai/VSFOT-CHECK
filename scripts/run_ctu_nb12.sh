#!/usr/bin/env bash
# =============================================================================
# 2026-10-28 Office-Home 12 任务：cap02_tstrong + 老师侧共识只在进度 < 0.25（--consensus teacher --ct_until 0.25），40 epoch，seed 42，串行
#   run_ctu_nb.sh 4 任务筛选：Ar2Cl 77.07（< 78.52，未过）、Pr2Cl 78.26、Re2Ar 85.05、Cl2Ar 84.86 过，VisDA 30% 88.44 过；4 任务和 +0.29。
#   筛选只差 Ar2Cl 一项，这里按最终规则直接看 12 任务平均是否 > cap02_tstrong 86.50（只看平均，不按任务挑开关，参数不再改）。
#   已跑的 4 个任务同目录同参数，自动跳过。
#
#   !bash scripts/run_ctu_nb12.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_ctu_nb12.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
GROUP="cap02_tstrong_ctu_nb"
EXTRA="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --consensus teacher --ct_until 0.25"

# 顺序：风险最大的 →Clipart 和落后 COSMO 的任务在前
PAIRS="art:clipart realworld:art art:product product:art product:clipart realworld:clipart \
clipart:product realworld:product art:realworld clipart:realworld product:realworld clipart:art"
abbr() { case "$1" in art) echo Ar ;; clipart) echo Cl ;; product) echo Pr ;; realworld) echo Re ;; esac; }
mkdir -p runs/ctu_nb12_logs

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    for pair in $PAIRS; do
        s="${pair%%:*}"; t="${pair##*:}"; task="$(abbr "$s")2$(abbr "$t")"
        dir="WeightSched/$GROUP/${task}_s42"
        if [ -f "runs/$dir/.done" ]; then echo "==> $task：已完成，跳过"; continue; fi
        if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
        echo "==> 开始 $task（$(date +%H:%M)）"
        # $EXTRA 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个任务
        if $PYTHON main.py --device "$GPU" --name "$dir" --dataset OfficeHomeDataset --source "$s" --target "$t" --seed 42 $EXTRA \
            > "runs/ctu_nb12_logs/${task}.log" 2>&1; then
            touch "runs/$dir/.done"; echo "==> 完成 $task"
        else
            echo "[错误] $task 训练失败，见 runs/ctu_nb12_logs/${task}.log"
        fi
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'EOF'
import os, re
cosmo = {"Ar2Cl": 76.4, "Ar2Pr": 92.6, "Ar2Re": 92.0, "Cl2Ar": 85.7, "Cl2Pr": 92.4, "Cl2Re": 91.9,
         "Pr2Ar": 84.3, "Pr2Cl": 76.1, "Pr2Re": 91.7, "Re2Ar": 85.6, "Re2Cl": 75.5, "Re2Pr": 92.7}
def evals(g, t):
    p = f"runs/WeightSched/{g}/{t}_s42/Detail.txt"
    if not os.path.exists(p) or not os.path.exists(f"runs/WeightSched/{g}/{t}_s42/.done"):
        return None
    return [float(x) * 100 for x in re.findall(r"Overall Accuracy \(OA\): ([0-9.]+)", open(p).read())][1:] or None
f = lambda v: "-" if v is None else f"{v:.2f}"
print("## Office-Home 12 任务：cap02_tstrong vs + 老师侧共识只在进度 < 0.25（--ct_until 0.25，40 epoch，seed 42，OA；后 5 = 最后 5 次评估均值）\n")
print("| 任务 | cap02_tstrong 后 5 | + ctu 后 5 | 差 | + ctu 最后一次 | COSMO | + ctu − COSMO |")
print("|---|---|---|---|---|---|---|")
a5, b5, bl = {}, {}, {}
for t in cosmo:
    a, b = evals("cap02_tstrong", t), evals("cap02_tstrong_ctu_nb", t)
    if a: a5[t] = sum(a[-5:]) / len(a[-5:])
    if b: b5[t] = sum(b[-5:]) / len(b[-5:]); bl[t] = b[-1]
    d = f"{b5[t] - a5[t]:+.2f}" if t in a5 and t in b5 else "-"
    dc = f"{b5[t] - cosmo[t]:+.2f}" if t in b5 else "-"
    print(f"| {t} | {f(a5.get(t))} | {f(b5.get(t))} | {d} | {f(bl.get(t))} | {cosmo[t]} | {dc} |")
done = [t for t in cosmo if t in a5 and t in b5]
if done:
    m = lambda d: sum(d[t] for t in done) / len(done)
    print(f"\n已完成 {len(done)} / 12；这些任务上平均：cap02_tstrong {m(a5):.2f}，+ ctu {m(b5):.2f}（最后一次 {m(bl):.2f}），"
          f"COSMO {sum(cosmo[t] for t in done) / len(done):.2f}")
    for dom in ("Cl", "Ar", "Pr", "Re"):
        ts = [t for t in done if t.endswith(dom)]
        if ts:
            print(f"→{dom}：差 {sum(b5[t] - a5[t] for t in ts) / len(ts):+.2f}，− COSMO {sum(b5[t] - cosmo[t] for t in ts) / len(ts):+.2f}（{len(ts)} 个任务）")
EOF
echo "============================== 汇总结束 =============================="
