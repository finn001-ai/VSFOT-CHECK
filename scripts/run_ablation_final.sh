#!/usr/bin/env bash
# =============================================================================
# 2026-10-31 消融（做实阶段）：按模块「从完整方法去掉一个」，seed 42，事先定好行，不按结果删行 / 改行
#   模块（按要解决的问题分组）：
#     A 可靠的对齐目标     --knn_k 10 --align_view both --align_weight cap --align_cap 0.2
#     B 老师的学习         --t_strong（VisDA / DN 另加 --consensus teacher）
#     C 目标域类别分布     --clsw_mode self（VisDA 另加 --align_norm_c 65）；OH 主配置不含 C
#     学生 EMA            --s_ema 0.5，只评估、不改训练，每行都开，不算模块
#   每行：完整 / −A / −B / −C / VSFOT（三个模块全去掉）
#   2026-10-31 gpuB 被抢占，全部在 gpuA 跑（torch 2.8）：源模型和「完整」行都用 gpuA 已有的（seed 42），同机可逐行比较
#   OH（ABL_SET=oh）：完整 = WeightSched/cap02_tstrong_sema；新跑 mod_minusA、mod_minusB、mod_vsfot（每行约 10.5 小时）
#     ABL_SET=oh_extra：mod_unified（OH 也开 C 和老师侧共识 = 三个数据集同一套组件，只作补充行，不替换主表）
#   VisDA（ABL_SET=visda）：完整 = VisDA/norm65_self_ct_e15_b32_sema；新跑 VisDA/mod_{minusA,minusB,minusC,vsfot}（每行约 2.1 小时）
#   DN（ABL_SET=dn）：完整 = DomainNet/cap02_tstrong_self_ct_sema_e10；新跑 DomainNet/mod_{minusC,minusB,minusA,vsfot}_e10（每行约 10 小时）
#   排队顺序（便宜、信息量大的先跑）：ABL_SET="visda oh dn oh_extra"
#   模块内部逐个组件的拆解放附录，用已有组（OH both_ramp / both_cap02 / cap02_tstrong，DN self_e10 / self_ct_e10）
#
#   !ABL_SET="visda oh dn oh_extra" bash scripts/run_ablation_final.sh
#   !SUMMARY_ONLY=1 bash scripts/run_ablation_final.sh    只看汇总（本机有的行）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
ABL_SET="${ABL_SET:-}"

A="--knn_k 10 --align_view both --align_weight cap --align_cap 0.2"
OH0="--epochs 40 --align_mode ot --s_ema 0.5"
VD0="--arch resnet101 --workers 12 --epochs 15 --batch_size 32 --align_mode ot --s_ema 0.5"
DN0="--arch resnet50 --workers 12 --epochs 10 --align_mode ot --s_ema 0.5"
VD_B="--t_strong --consensus teacher"; VD_C="--clsw_mode self --align_norm_c 65"
DN_B="--t_strong --consensus teacher"; DN_C="--clsw_mode self"

OH_T="Ar:art:Cl:clipart Ar:art:Pr:product Ar:art:Re:realworld Cl:clipart:Ar:art Cl:clipart:Pr:product Cl:clipart:Re:realworld \
Pr:product:Ar:art Pr:product:Cl:clipart Pr:product:Re:realworld Re:realworld:Ar:art Re:realworld:Cl:clipart Re:realworld:Pr:product"
DN_T="real:sketch real:clipart painting:sketch clipart:sketch sketch:clipart painting:clipart \
real:painting clipart:painting sketch:painting painting:real clipart:real sketch:real"

# 格式 "目录|数据集|源域|目标域|源模型后缀|参数"
jobs_for() {
    case "$1" in
        oh|oh_extra)
            if [ "$1" = oh ]; then gs=("mod_minusA|$OH0 --t_strong" "mod_minusB|$OH0 $A" "mod_vsfot|$OH0")
            else gs=("mod_unified|$OH0 $A --t_strong --consensus teacher --clsw_mode self --align_norm_c 65"); fi
            for g in "${gs[@]}"; do
                for p in $OH_T; do IFS=: read -r a s b t <<< "$p"
                    echo "WeightSched/${g%%|*}/${a}2${b}_s42|OfficeHomeDataset|$s|$t||${g#*|}"; done; done ;;
        visda) for g in "minusA|$VD_B $VD_C" "minusB|$A $VD_C" "minusC|$A $VD_B" "vsfot|"; do
                   echo "VisDA/mod_${g%%|*}/S2R_s42|VisDA|synthesis|real||$VD0 ${g#*|}"; done ;;
        dn) for g in "minusC|$A $DN_B" "minusB|$A $DN_C" "minusA|$DN_B $DN_C" "vsfot|"; do
                for p in $DN_T; do IFS=: read -r s t <<< "$p"
                    a=$(echo "$s" | cut -c1 | tr a-z A-Z); b=$(echo "$t" | cut -c1 | tr a-z A-Z)
                    echo "DomainNet/mod_${g%%|*}_e10/${a}2${b}_s42|DomainNet|$s|$t||$DN0 ${g#*|}"; done; done ;;
        *) echo "未知 ABL_SET: $1" >&2; exit 1 ;;
    esac
}

if [ "$SUMMARY_ONLY" != "1" ]; then
    [ -n "$ABL_SET" ] || { echo "[错误] 需要 ABL_SET=visda / oh / dn / oh_extra（可空格分隔多个）"; exit 1; }
    mkdir -p runs/ablation_final_logs
    echo "机器 $(hostname)，GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)，torch $($PYTHON -c 'import torch; print(torch.__version__)')，commit $(git log -1 --format=%h)，ABL_SET=$ABL_SET"
    T0=$(date +%s)
    for set in $ABL_SET; do
        while IFS= read -r item; do
            IFS="|" read -r dir ds s t suf extra <<< "$item"
            if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
            if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
            echo "==> 开始 $dir（$(date +%H:%M)）  参数：$extra"
            # $extra 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个
            if $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 --src_suffix "$suf" $extra \
                > "runs/ablation_final_logs/$(echo "$dir" | tr / _).log" 2>&1; then
                touch "runs/$dir/.done"; echo "==> 完成 $dir"
            else
                echo "[错误] $dir 训练失败，见 runs/ablation_final_logs/"
            fi
        done < <(jobs_for "$set")
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'PY'
import os, re
def done(d): return os.path.exists(f"runs/{d}/.done") or (os.path.exists(f"runs/{d}/Detail.txt") and "/mod_" not in d)
def detail(d, key):
    p = f"runs/{d}/Detail.txt"
    if not done(d) or not os.path.exists(p):
        return None
    return [float(x) * 100 for x in re.findall(key + r": ([0-9.]+)", open(p).read())][1:] or None
def ema(d, i):
    p = f"runs/{d}/run_log.txt"
    if not done(d) or not os.path.exists(p):
        return None
    return [float(x[i]) * 100 for x in re.findall(r"\[ema\] iter \d+  OA ([0-9.]+)  AA ([0-9.]+)", open(p).read())] or None
m5 = lambda v: None if not v else sum(v[-5:]) / len(v[-5:])
f = lambda v: "-" if v is None else f"{v:.2f}"
OA, AA = "Overall Accuracy \\(OA\\)", "Average Accuracy \\(AA\\)"
avg = lambda xs: None if not xs or None in xs else sum(xs) / len(xs)

def multi(title, root, tasks, rows, groups_by_tgt):
    print(f"## {title}（seed 42；{len(tasks)} 任务 OA 后 5 次均值的平均；EMA 学生，括号内为原学生）\n")
    print("| 行 | 组 | EMA（原学生） | " + " | ".join(groups_by_tgt) + " | 完成任务 |")
    print("|---|---|---|" + "---|" * len(groups_by_tgt) + "---|")
    for name, g in rows:
        r = [m5(detail(f"{root}/{g}/{t}_s42", OA)) for t in tasks]
        e = [m5(ema(f"{root}/{g}/{t}_s42", 0)) for t in tasks]
        tgt = lambda k: f(avg([v for t, v in zip(tasks, e) if t.endswith(k)]))
        print(f"| {name} | {g} | {f(avg(e))}（{f(avg(r))}） | " + " | ".join(tgt(k[-2:].strip("→")) for k in groups_by_tgt)
              + f" | {sum(x is not None for x in e)} / {len(tasks)} |")
    print()

multi("Office-Home 模块消融", "WeightSched",
      ["Ar2Cl", "Ar2Pr", "Ar2Re", "Cl2Ar", "Cl2Pr", "Cl2Re", "Pr2Ar", "Pr2Cl", "Pr2Re", "Re2Ar", "Re2Cl", "Re2Pr"],
      [("完整（A + B，主表）", "cap02_tstrong_sema"), ("− A 对齐目标", "mod_minusA"), ("− B 老师学习", "mod_minusB"),
       ("VSFOT（全去掉）", "mod_vsfot"), ("补充：+ C + 老师侧共识（三数据集同一套组件）", "mod_unified")],
      ["→Cl", "→Ar", "→Pr", "→Re"])

print("## VisDA-C 模块消融（seed 42，gpuA；后 5 次均值）\n")
print("| 行 | EMA AA（原学生） | EMA OA（原学生） |")
print("|---|---|---|")
for name, g in [("完整（主表）", "norm65_self_ct_e15_b32_sema"), ("− A 对齐目标", "mod_minusA"), ("− B 老师学习", "mod_minusB"), ("− C 类别分布", "mod_minusC"), ("VSFOT（全去掉）", "mod_vsfot")]:
    d = f"VisDA/{g}/S2R_s42"
    print(f"| {name} | {f(m5(ema(d, 1)))}（{f(m5(detail(d, AA)))}） | {f(m5(ema(d, 0)))}（{f(m5(detail(d, OA)))}） |")
print()

multi("DomainNet-126 模块消融", "DomainNet",
      ["R2S", "R2C", "P2S", "C2S", "S2C", "P2C", "R2P", "C2P", "S2P", "P2R", "C2R", "S2R"],
      [("完整（主表）", "cap02_tstrong_self_ct_sema_e10"), ("− A 对齐目标", "mod_minusA_e10"), ("− B 老师学习", "mod_minusB_e10"),
       ("− C 类别分布", "mod_minusC_e10"), ("VSFOT（全去掉）", "mod_vsfot_e10")],
      ["→S", "→C", "→P", "→R"])
PY
echo "============================== 汇总结束 =============================="
