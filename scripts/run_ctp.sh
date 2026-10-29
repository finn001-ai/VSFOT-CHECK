#!/usr/bin/env bash
# =============================================================================
# 2026-10-29 老师拆成两路（--t_prompt --ct_prompt）：adapter 照原方法学学生 top-3，提示上下文学老师 × 学生专家乘积共识，seed 42，串行
#   run_ctp.sh（只学提示 + 共识）：没有整类被吞（Pr2Cl Clipboards 80 vs 全程共识 0，Ar2Cl Marker 37 vs 6），但去掉 adapter 老师太弱
#   （Cl2Ar 80.48、Re2Ar 80.87，无共识对照 Cl2Ar 80.60 —— 掉点来自去掉 adapter，不是共识）。这里保留 adapter 的原学习信号，共识只经低容量的
#   共享上下文（4 个向量，Adam 1e-5，已有默认）作用到类别层面。
#   判据（事先定）：Ar2Cl / Pr2Cl ≥ 78.52 / 78.10，Re2Ar / Cl2Ar ≥ 84.56 / 83.77，VisDA 30% ≥ 88.56（VisDA 现配置 = adapter 老师 + 共识；
#   要三个数据集统一用这一种老师，VisDA 也得不低于现配置）；全过才跑 OH 剩余 8 个任务，否则停。
#
#   !bash scripts/run_ctp.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_ctp.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

OH="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --t_prompt --ct_prompt --log_dynamics --log_samples"
VD="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 3 --batch_size 32 --data_frac 0.3 --clsw_mode self --align_norm_c 65 --t_prompt --ct_prompt"

# 格式 "目录|数据集|源域|目标域|参数"
JOBS=(
    "WeightSched/cap02_tstrong_ctp/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH"
    "WeightSched/cap02_tstrong_ctp/Re2Ar_s42|OfficeHomeDataset|realworld|art|$OH"
    "VisDA/q30_norm65_ctp/S2R_s42|VisDA|synthesis|real|$VD"
    "WeightSched/cap02_tstrong_ctp/Cl2Ar_s42|OfficeHomeDataset|clipart|art|$OH"
    "WeightSched/cap02_tstrong_ctp/Pr2Cl_s42|OfficeHomeDataset|product|clipart|$OH"
)
mkdir -p runs/ctp_logs

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    for item in "${JOBS[@]}"; do
        IFS="|" read -r dir ds s t extra <<< "$item"
        if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
        if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
        echo "==> 开始 $dir（$(date +%H:%M)）"
        # $extra 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个
        if $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 $extra \
            > "runs/ctp_logs/$(echo "$dir" | tr / _).log" 2>&1; then
            touch "runs/$dir/.done"; echo "==> 完成 $dir"
        else
            echo "[错误] $dir 训练失败，见 runs/ctp_logs/"
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
print("| 任务 | cap02_tstrong | + 共识 | + ct_until 0.25 | 只学提示 + 共识 | + 两路老师 后 5 | 最后一次 | COSMO |")
print("|---|---|---|---|---|---|---|---|")
cosmo = {"Ar2Cl": 76.4, "Re2Ar": 85.6, "Cl2Ar": 85.7, "Pr2Cl": 76.1}
for t in cosmo:
    r = []
    for g in ("cap02_tstrong", "cap02_tstrong_ct", "cap02_tstrong_ctu_nb", "cap02_tstrong_pc", "cap02_tstrong_ctp"):
        v = vals(f"WeightSched/{g}/{t}_s42", OA)
        r.append((sum(v[-5:]) / len(v[-5:]), v[-1]) if v else (None, None))
    print(f"| {t} | {f(r[0][0])} | {f(r[1][0])} | {f(r[2][0])} | {f(r[3][0])} | {f(r[4][0])} | {f(r[4][1])} | {cosmo[t]} |")
print("\n## VisDA-C 30%（AA 均值 = 不含第 0 步的全部评估 / 最后一次）\n")
for g in ("q30_norm65", "q30_norm65_cons_t", "q30_norm65_pc", "q30_norm65_ctp"):
    v = vals(f"VisDA/{g}/S2R_s42", AA)
    print(f"{g}: {f(sum(v) / len(v) if v else None)} / {f(v[-1] if v else None)}")
    p = f"runs/VisDA/{g}/S2R_s42/ModellineAcc.txt"
    if os.path.exists(p):
        L = open(p).read().strip().splitlines(); print("  ", L[0]); print("  ", L[-1])
print("\n## 被吞的类（最后一次评估准确率，无共识 / 共识 / ctm）")
for t, cls in (("Ar2Cl", ["Computer", "Pencil", "Marker", "Monitor", "Pen"]), ("Pr2Cl", ["Clipboards", "Pencil", "Folder", "Pen"]), ("Re2Ar", ["Monitor", "Computer"])):
    row = []
    for g in ("cap02_tstrong", "cap02_tstrong_ct", "cap02_tstrong_ctp"):
        p = f"runs/WeightSched/{g}/{t}_s42/ModellineAcc.txt"
        if not os.path.exists(p):
            row.append(None); continue
        L = open(p).read().rstrip("\n").splitlines()
        names = [L[0][i:i + 12].strip() for i in range(12, len(L[0]), 12)]
        row.append(dict(zip(names, map(float, L[-1].split()[1:]))))
    if all(row):
        print(t + "：" + "，".join(f"{c} {row[0][c] * 100:.0f} / {row[1][c] * 100:.0f} / {row[2][c] * 100:.0f}" for c in cls) + "（无共识 / 共识 / 两路老师）")
PY
echo "============================== 汇总结束 =============================="
