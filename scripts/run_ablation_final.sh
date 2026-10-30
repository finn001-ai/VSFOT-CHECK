#!/usr/bin/env bash
# =============================================================================
# 2026-10-31 消融（做实阶段）：逐步加组件的累积表，seed 42，事先定好行，不按结果删行 / 改行
#   组件顺序（与方法描述一致）：
#     R0 VSFOT 原方法（OT 对齐，ramp 权重，weak 视图）
#     R1 + kNN10 平滑                 --knn_k 10
#     R2 + 双视图对齐                  --align_view both
#     R3 + cap 权重                    --align_weight cap --align_cap 0.2
#     R4 + 老师强增强                  --t_strong
#     R5 + 按类别数归一化（VisDA）      --align_norm_c 65
#     R6 + 自估目标分布（VisDA / DN）   --clsw_mode self
#     R7 + 老师侧共识（VisDA / DN）     --consensus teacher
#     + 学生 EMA                       --s_ema 0.5（只评估、不改训练：每行都开，EMA 列免费得到）
#   OH（ABL_SET=oh，必须在 gpuA 跑：R2–R5 已有结果都在 gpuA，torch 2.8，同一台机器才可逐行比较）
#     只缺 R0、R1：WeightSched/abl_r0_vsfot、WeightSched/abl_r1_knn，12 任务，40 epoch，seed 42 源模型
#     已有：R2 both_ramp，R3 both_cap02，R4 cap02_tstrong，R4 + EMA cap02_tstrong_sema
#   VisDA（ABL_SET=visda，在 gpuB 跑整条链，含最终配置 R7，表内自洽；源模型在本机用 seed 42 训练，放 runs/Source_visda_b42）
#     VisDA/abl_b42_r0 … r7，ResNet-101，15 epoch，batch 32；先跑 r7（核对与 gpuA 主表 92.33 的差距），再 r0 → r6
#   DN 不新跑：已有 R4 cap02_tstrong_e10，R6 cap02_tstrong_self_e10，R7 cap02_tstrong_self_ct_e10，+ EMA self_ct_sema（都在 gpuA）
#
#   !ABL_SET=oh bash scripts/run_ablation_final.sh        !ABL_SET=visda bash scripts/run_ablation_final.sh
#   !SUMMARY_ONLY=1 bash scripts/run_ablation_final.sh    只看汇总（本机有的行）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
ABL_SET="${ABL_SET:-}"

OH0="--epochs 40 --align_mode ot --s_ema 0.5"
VD0="--arch resnet101 --workers 12 --epochs 15 --batch_size 32 --align_mode ot --s_ema 0.5"
VD_STEPS=("" "--knn_k 10" "--align_view both" "--align_weight cap --align_cap 0.2" "--t_strong" "--align_norm_c 65" "--clsw_mode self" "--consensus teacher")

# 格式 "目录|数据集|源域|目标域|源模型后缀|参数"
jobs_for() {
    case "$1" in
        oh) for g in "abl_r0_vsfot|$OH0" "abl_r1_knn|$OH0 --knn_k 10"; do
                for p in Ar:art:Cl:clipart Ar:art:Pr:product Ar:art:Re:realworld Cl:clipart:Ar:art Cl:clipart:Pr:product Cl:clipart:Re:realworld \
                         Pr:product:Ar:art Pr:product:Cl:clipart Pr:product:Re:realworld Re:realworld:Ar:art Re:realworld:Cl:clipart Re:realworld:Pr:product; do
                    IFS=: read -r a s b t <<< "$p"; echo "WeightSched/${g%%|*}/${a}2${b}_s42|OfficeHomeDataset|$s|$t||${g#*|}"; done; done ;;
        visda) for r in 7 0 1 2 3 4 5 6; do
                   extra="$VD0"; for k in $(seq 1 "$r"); do extra="$extra ${VD_STEPS[$k]}"; done
                   echo "VisDA/abl_b42_r$r/S2R_s42|VisDA|synthesis|real|_b42|$extra"; done ;;
        *) echo "未知 ABL_SET: $1" >&2; exit 1 ;;
    esac
}

if [ "$SUMMARY_ONLY" != "1" ]; then
    [ -n "$ABL_SET" ] || { echo "[错误] 需要 ABL_SET=oh 或 visda"; exit 1; }
    mkdir -p runs/ablation_final_logs
    echo "机器 $(hostname)，GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)，torch $($PYTHON -c 'import torch; print(torch.__version__)')，commit $(git log -1 --format=%h)，ABL_SET=$ABL_SET"
    T0=$(date +%s)
    if [ "$ABL_SET" = visda ] && [ ! -f runs/Source_visda_b42/Source_synthesis/Source.pt ]; then
        [ -d runs/Source_visda_b42/Source_synthesis ] && { echo "[错误] runs/Source_visda_b42/Source_synthesis 已存在但没有 Source.pt"; exit 1; }
        echo "==> 训练源模型 runs/Source_visda_b42/Source_synthesis（seed 42，参数同 gpuA 的 Source_visda）（$(date +%H:%M)）"
        $PYTHON trainSource.py --name Source_visda_b42/Source_synthesis --seed 42 --dataset VisDA --source synthesis --arch resnet101 --epochs 2 --lr 0.001 \
            > runs/ablation_final_logs/Source_visda_b42.log 2>&1 || { echo "[错误] 源模型训练失败"; exit 1; }
    fi
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
    done < <(jobs_for "$ABL_SET")
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'PY'
import os, re
def done(d): return os.path.exists(f"runs/{d}/.done") or (os.path.exists(f"runs/{d}/Detail.txt") and not d.startswith(("WeightSched/abl_", "VisDA/abl_")))
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

print("## Office-Home 累积消融（seed 42，gpuA；12 任务 OA 后 5 次均值的平均）\n")
T = ["Ar2Cl", "Ar2Pr", "Ar2Re", "Cl2Ar", "Cl2Pr", "Cl2Re", "Pr2Ar", "Pr2Cl", "Pr2Re", "Re2Ar", "Re2Cl", "Re2Pr"]
rows = [("R0 VSFOT", "abl_r0_vsfot"), ("R1 + kNN10", "abl_r1_knn"), ("R2 + 双视图", "both_ramp"), ("R3 + cap 权重", "both_cap02"),
        ("R4 + 老师强增强", "cap02_tstrong"), ("R4 + 学生 EMA（最终）", "cap02_tstrong_sema")]
print("| 行 | 组 | 原学生 | EMA | →Cl | →Ar | →Pr | →Re | 完成任务 |")
print("|---|---|---|---|---|---|---|---|---|")
for name, g in rows:
    r = [m5(detail(f"WeightSched/{g}/{t}_s42", OA)) for t in T]
    e = [m5(ema(f"WeightSched/{g}/{t}_s42", 0)) for t in T]
    use = e if g == "cap02_tstrong_sema" else r
    tgt = lambda k: f(avg([v for t, v in zip(T, use) if t.endswith(k)]))
    print(f"| {name} | {g} | {f(avg(r))} | {f(avg(e)) if any(x is not None for x in e) else '-'} | {tgt('Cl')} | {tgt('Ar')} | {tgt('Pr')} | {tgt('Re')} | {sum(x is not None for x in r)} / 12 |")

print("\n## VisDA-C 累积消融（seed 42，gpuB 整条链，源模型 Source_visda_b42；AA 后 5 次均值）\n")
names = ["R0 VSFOT", "R1 + kNN10", "R2 + 双视图", "R3 + cap 权重", "R4 + 老师强增强", "R5 + 类别数归一化", "R6 + 自估目标分布", "R7 + 老师侧共识（最终）"]
print("| 行 | 原学生 AA | EMA AA | 原学生 OA | EMA OA |")
print("|---|---|---|---|---|")
for r, name in enumerate(names):
    d = f"VisDA/abl_b42_r{r}/S2R_s42"
    print(f"| {name} | {f(m5(detail(d, AA)))} | {f(m5(ema(d, 1)))} | {f(m5(detail(d, OA)))} | {f(m5(ema(d, 0)))} |")

print("\n## DomainNet-126（seed 42，gpuA，已有组；12 任务 OA 后 5 次均值的平均）\n")
DT = ["R2S", "R2C", "P2S", "C2S", "S2C", "P2C", "R2P", "C2P", "S2P", "P2R", "C2R", "S2R"]
print("| 行 | 组 | 原学生 | EMA | 完成任务 |")
print("|---|---|---|---|---|")
for name, g in [("R4", "cap02_tstrong_e10"), ("R6 + 自估目标分布", "cap02_tstrong_self_e10"), ("R7 + 老师侧共识", "cap02_tstrong_self_ct_e10"), ("R7 + 学生 EMA（最终）", "cap02_tstrong_self_ct_sema_e10")]:
    r = [m5(detail(f"DomainNet/{g}/{t}_s42", OA)) for t in DT]
    e = [m5(ema(f"DomainNet/{g}/{t}_s42", 0)) for t in DT]
    print(f"| {name} | {g} | {f(avg(r))} | {f(avg(e)) if any(x is not None for x in e) else '-'} | {sum(x is not None for x in r)} / 12 |")
PY
echo "============================== 汇总结束 =============================="
