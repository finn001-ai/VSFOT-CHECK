#!/usr/bin/env bash
# =============================================================================
# 2026-10-28 老师侧共识 + 类均衡，只在早期使用（--consensus teacher --ct_bal --ct_until 0.25），seed 42，串行
#   run_ctbal.sh：--ct_bal 修好了整类塌缩，早期提升更大（Ar2Cl 第 2 次评估 +4.0），但后期在学生领先的 →Cl 上仍转负（−1.3），Re2Ar 丢掉收益。
#   全部任务上共识的收益集中在早期（第 2–4 次评估 +0.7 ~ +2.6），交叉点约在进度 7.5% ~ 12.5%；切换点沿用已有的 0.25（self 先验冻结点），不另调。
#   判据（事先定）：Ar2Cl / Pr2Cl ≥ cap02_tstrong（78.52 / 78.10），Re2Ar / Cl2Ar ≥ cap02_tstrong（84.56 / 83.77），VisDA 30% ≥ q30_norm65（88.06）。
#   通过后跑 OH 剩余 7 个任务，看 12 任务平均是否 > 86.50。
#
#   !bash scripts/run_ctu.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_ctu.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

OH="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --consensus teacher --ct_bal --ct_until 0.25 --log_dynamics --log_samples"
VD="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 3 --batch_size 32 --data_frac 0.3 --clsw_mode self --align_norm_c 65 --consensus teacher --ct_bal --ct_until 0.25"

# 格式 "目录|数据集|源域|目标域|参数"
JOBS=(
    "WeightSched/cap02_tstrong_ctu/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH"
    "WeightSched/cap02_tstrong_ctu/Re2Ar_s42|OfficeHomeDataset|realworld|art|$OH"
    "VisDA/q30_norm65_cons_t_bal_u/S2R_s42|VisDA|synthesis|real|$VD"
    "WeightSched/cap02_tstrong_ctu/Cl2Ar_s42|OfficeHomeDataset|clipart|art|$OH"
    "WeightSched/cap02_tstrong_ctu/Pr2Cl_s42|OfficeHomeDataset|product|clipart|$OH"
)
mkdir -p runs/ctu_logs

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    for item in "${JOBS[@]}"; do
        IFS="|" read -r dir ds s t extra <<< "$item"
        if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
        if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
        echo "==> 开始 $dir（$(date +%H:%M)）"
        # $extra 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个
        if $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 $extra \
            > "runs/ctu_logs/$(echo "$dir" | tr / _).log" 2>&1; then
            touch "runs/$dir/.done"; echo "==> 完成 $dir"
        else
            echo "[错误] $dir 训练失败，见 runs/ctu_logs/"
        fi
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'PY'
import os, re
def vals(d, key):
    p = f"runs/{d}/Detail.txt"
    if not os.path.exists(p) or not os.path.exists(f"runs/{d}/.done"):
        return None
    return [float(x) * 100 for x in re.findall(key + r": ([0-9.]+)", open(p).read())][1:] or None
f = lambda v: "-" if v is None else f"{v:.2f}"
OA, AA = "Overall Accuracy \\(OA\\)", "Average Accuracy \\(AA\\)"
print("## Office-Home（OA 后 5 / 最后一次）\n")
print("| 任务 | cap02_tstrong | + 共识 | + ct_bal | + ct_bal + ct_until 0.25 后 5 | 最后一次 | COSMO |")
print("|---|---|---|---|---|---|---|")
cosmo = {"Ar2Cl": 76.4, "Re2Ar": 85.6, "Cl2Ar": 85.7, "Pr2Cl": 76.1}
for t in cosmo:
    r = []
    for g in ("cap02_tstrong", "cap02_tstrong_ct", "cap02_tstrong_ctbal", "cap02_tstrong_ctu"):
        v = vals(f"WeightSched/{g}/{t}_s42", OA)
        r.append((sum(v[-5:]) / len(v[-5:]), v[-1]) if v else (None, None))
    print(f"| {t} | {f(r[0][0])} | {f(r[1][0])} | {f(r[2][0])} | {f(r[3][0])} | {f(r[3][1])} | {cosmo[t]} |")
print("\n## VisDA-C 30%（AA 均值 = 不含第 0 步的全部评估 / 最后一次）\n")
for g in ("q30_norm65", "q30_norm65_cons_t", "q30_norm65_cons_t_bal", "q30_norm65_cons_t_bal_u"):
    v = vals(f"VisDA/{g}/S2R_s42", AA)
    print(f"{g}: {f(sum(v) / len(v) if v else None)} / {f(v[-1] if v else None)}")
    p = f"runs/VisDA/{g}/S2R_s42/ModellineAcc.txt"
    if os.path.exists(p):
        L = open(p).read().strip().splitlines(); print("  ", L[0]); print("  ", L[-1])
print("\n## 被吞的类（最后一次评估准确率，无共识 / 共识 / ctu）")
for t, cls in (("Ar2Cl", ["Computer", "Pencil", "Marker", "Monitor", "Pen"]), ("Pr2Cl", ["Clipboards", "Pencil", "Folder", "Pen"]), ("Re2Ar", ["Monitor", "Computer"])):
    row = []
    for g in ("cap02_tstrong", "cap02_tstrong_ct", "cap02_tstrong_ctu"):
        p = f"runs/WeightSched/{g}/{t}_s42/ModellineAcc.txt"
        if not os.path.exists(p):
            row.append(None); continue
        L = open(p).read().rstrip("\n").splitlines()
        names = [L[0][i:i + 12].strip() for i in range(12, len(L[0]), 12)]
        row.append(dict(zip(names, map(float, L[-1].split()[1:]))))
    if all(row):
        print(t + "：" + "，".join(f"{c} {row[0][c] * 100:.0f} / {row[1][c] * 100:.0f} / {row[2][c] * 100:.0f}" for c in cls) + "（无共识 / 共识 / ctu）")
PY
echo "============================== 汇总结束 =============================="
