#!/usr/bin/env bash
# =============================================================================
# 2026-10-29 学生 EMA（--s_ema 0.5：学生权重 + BN 统计量的指数滑动平均，时间常数 0.5 个 epoch，事先定、不调），seed 42，串行
#   动机：VisDA 主配置两次全量都在 iter 24220 出现 train 类瞬时塌陷（1595 张 → knife，AA 92.3 → 89.0 → 92.3），
#   BN 重估能完全救回（后 5 次 91.60 → 92.35），但 OH 上 BN 重估有害（Ar2Re −0.22，换了 BN 统计量的来源）。
#   EMA 只平滑时间上的抖动、不换统计量来源；只额外评估，训练与主配置逐位相同（原指标列应精确复现）。
#   判据（事先定）：
#     OH 4 个任务 EMA 后 5 次均值的平均 ≥ 原学生的平均，且每个任务 ≥ 原学生 − 0.2
#     VisDA 全量 EMA 后 5 次 AA ≥ 92.0（原 91.60）
#     原指标列与主配置不一致（不可复现）则结果作废
#   OH 通过才跑 OH 剩余 8 个任务。
#
#   !bash scripts/run_sema.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_sema.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

OH="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --s_ema 0.5"
VD="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 15 --batch_size 32 --clsw_mode self --align_norm_c 65 --consensus teacher --s_ema 0.5"

# 格式 "目录|数据集|源域|目标域|参数"
JOBS=(
    "WeightSched/cap02_tstrong_sema/Ar2Cl_s42|OfficeHomeDataset|art|clipart|$OH"
    "WeightSched/cap02_tstrong_sema/Re2Ar_s42|OfficeHomeDataset|realworld|art|$OH"
    "WeightSched/cap02_tstrong_sema/Cl2Ar_s42|OfficeHomeDataset|clipart|art|$OH"
    "WeightSched/cap02_tstrong_sema/Pr2Cl_s42|OfficeHomeDataset|product|clipart|$OH"
    "VisDA/norm65_self_ct_e15_b32_sema/S2R_s42|VisDA|synthesis|real|$VD"
)
mkdir -p runs/sema_logs

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    for item in "${JOBS[@]}"; do
        IFS="|" read -r dir ds s t extra <<< "$item"
        if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
        if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
        echo "==> 开始 $dir（$(date +%H:%M)）"
        # $extra 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个
        if $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 $extra \
            > "runs/sema_logs/$(echo "$dir" | tr / _).log" 2>&1; then
            touch "runs/$dir/.done"; echo "==> 完成 $dir"
        else
            echo "[错误] $dir 训练失败，见 runs/sema_logs/"
        fi
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'PY'
import os, re
def detail(d, key):
    p = f"runs/{d}/Detail.txt"
    if not os.path.exists(p):
        return None
    return [float(x) * 100 for x in re.findall(key + r": ([0-9.]+)", open(p).read())][1:] or None
def ema(d, key):
    p = f"runs/{d}/run_log.txt"
    if not os.path.exists(p):
        return None
    return [float(x) * 100 for x in re.findall(r"\[ema\] iter \d+  OA ([0-9.]+)  AA ([0-9.]+)", open(p).read()) for x in [x[0 if key == "OA" else 1]]] or None
m5 = lambda v: None if not v else sum(v[-5:]) / len(v[-5:])
f = lambda v: "-" if v is None else f"{v:.2f}"
OA, AA = "Overall Accuracy \\(OA\\)", "Average Accuracy \\(AA\\)"
print("## Office-Home（OA 后 5 次均值）\n")
print("| 任务 | 主配置 cap02_tstrong | 本次原学生（应相同） | 本次 EMA | EMA − 原 | 完成 |")
print("|---|---|---|---|---|---|")
ok, dr, de = True, [], []
for t in ("Ar2Cl", "Re2Ar", "Cl2Ar", "Pr2Cl"):
    d = f"WeightSched/cap02_tstrong_sema/{t}_s42"
    b, r, e = m5(detail(f"WeightSched/cap02_tstrong/{t}_s42", OA)), m5(detail(d, OA)), m5(ema(d, "OA"))
    done = os.path.exists(f"runs/{d}/.done")
    if r is not None and e is not None and done:
        dr.append(r); de.append(e); ok &= e >= r - 0.2
    print(f"| {t} | {f(b)} | {f(r)} | {f(e)} | {'-' if r is None or e is None else f'{e - r:+.2f}'} | {'是' if done else '否'} |")
if len(dr) == 4:
    print(f"\n平均：原 {sum(dr) / 4:.2f}，EMA {sum(de) / 4:.2f}；判据 {'过' if ok and sum(de) >= sum(dr) else '不过'}")
print("\n## VisDA-C 全量（AA 后 5 次均值 / 最后一次）\n")
d = "VisDA/norm65_self_ct_e15_b32_sema/S2R_s42"
for name, v in (("主配置", detail("VisDA/norm65_self_ct_e15_b32/S2R_s42", AA)), ("本次原学生", detail(d, AA)), ("本次 EMA", ema(d, "AA"))):
    print(f"{name}: {f(m5(v))} / {f(v[-1] if v else None)}；逐次 {' '.join(f'{x:.2f}' for x in v[-6:]) if v else '-'}")
PY
echo "============================== 汇总结束 =============================="
