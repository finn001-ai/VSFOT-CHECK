#!/usr/bin/env bash
# =============================================================================
# 2026-10-30 最终配置的多种子（做实阶段，不再加新组件）
#   种子事先定：42（已有）、0、1；不挑种子。结果放在 seed 42 的同一组目录下，只换 _s<seed> 后缀
#   最终配置（与 seed 42 主表相同）：
#     OH     WeightSched/cap02_tstrong_sema         40 epoch，cap 0.2，tstrong，s_ema 0.5
#     VisDA  VisDA/norm65_self_ct_e15_b32_sema     ResNet-101，15 epoch，batch 32，clsw self，norm 65，老师侧共识，s_ema 0.5
#     DN     DomainNet/cap02_tstrong_self_ct_sema_e10   ResNet-50，10 epoch，clsw self，老师侧共识，s_ema 0.5
#   每个任务结束保存 final.pt（main.py 默认 --save_final 1）
#
#   !SEED=0 SETS="oh" bash scripts/run_seeds_final.sh          SETS 可选 oh / visda / dn，空格分隔，按给出的顺序跑
#   !SUMMARY_ONLY=1 bash scripts/run_seeds_final.sh            只看汇总（本机上已有的所有种子）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
SEED="${SEED:-}"
SETS="${SETS:-oh visda dn}"

OH="--epochs 40 --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong --s_ema 0.5"
VD="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 15 --batch_size 32 --clsw_mode self --align_norm_c 65 --consensus teacher --s_ema 0.5"
DN="--align_mode ot --knn_k 10 --align_view both --arch resnet50 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 10 --clsw_mode self --consensus teacher --s_ema 0.5"

OH_G="WeightSched/cap02_tstrong_sema"
VD_G="VisDA/norm65_self_ct_e15_b32_sema"
DN_G="DomainNet/cap02_tstrong_self_ct_sema_e10"

# 格式 "目录（不含 _s<seed>）|数据集|源域|目标域|参数"
jobs_for() {
    case "$1" in
        oh) for p in Ar:art:Cl:clipart Ar:art:Pr:product Ar:art:Re:realworld Cl:clipart:Ar:art Cl:clipart:Pr:product Cl:clipart:Re:realworld \
                     Pr:product:Ar:art Pr:product:Cl:clipart Pr:product:Re:realworld Re:realworld:Ar:art Re:realworld:Cl:clipart Re:realworld:Pr:product; do
                IFS=: read -r a s b t <<< "$p"; echo "$OH_G/${a}2${b}|OfficeHomeDataset|$s|$t|$OH"; done ;;
        visda) echo "$VD_G/S2R|VisDA|synthesis|real|$VD" ;;
        dn) for p in real:sketch real:clipart painting:sketch clipart:sketch sketch:clipart painting:clipart \
                     real:painting clipart:painting sketch:painting painting:real clipart:real sketch:real; do
                IFS=: read -r s t <<< "$p"; a=$(echo "$s" | cut -c1 | tr a-z A-Z); b=$(echo "$t" | cut -c1 | tr a-z A-Z)
                echo "$DN_G/${a}2${b}|DomainNet|$s|$t|$DN"; done ;;
        *) echo "未知数据集 $1" >&2; exit 1 ;;
    esac
}

if [ "$SUMMARY_ONLY" != "1" ]; then
    [ -n "$SEED" ] || { echo "[错误] 需要 SEED=<种子>"; exit 1; }
    mkdir -p runs/seeds_final_logs
    echo "机器 $(hostname)，GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)，commit $(git log -1 --format=%h)，SEED=$SEED，SETS=$SETS"
    T0=$(date +%s)
    for set in $SETS; do
        while IFS= read -r item; do
            IFS="|" read -r base ds s t extra <<< "$item"
            dir="${base}_s${SEED}"
            if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; continue; fi
            if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; continue; fi
            echo "==> 开始 $dir（$(date +%H:%M)）"
            # $extra 不加引号，让它拆成多个参数；失败不写完成标记，继续下一个
            if $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed "$SEED" $extra \
                > "runs/seeds_final_logs/$(echo "$dir" | tr / _).log" 2>&1; then
                touch "runs/$dir/.done"; echo "==> 完成 $dir"
            else
                echo "[错误] $dir 训练失败，见 runs/seeds_final_logs/"
            fi
        done < <(jobs_for "$set")
    done
    echo "耗时 $(( ($(date +%s) - T0) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
OH_G="$OH_G" VD_G="$VD_G" DN_G="$DN_G" $PYTHON - <<'PY'
import os, re, statistics as st
SEEDS = [42, 0, 1]
def detail(d, key):
    p = f"runs/{d}/Detail.txt"
    if not os.path.exists(f"runs/{d}/.done") or not os.path.exists(p):
        return None
    return [float(x) * 100 for x in re.findall(key + r": ([0-9.]+)", open(p).read())][1:] or None
def ema(d, key):
    p = f"runs/{d}/run_log.txt"
    if not os.path.exists(f"runs/{d}/.done") or not os.path.exists(p):
        return None
    return [float(x[0 if key == "OA" else 1]) * 100 for x in re.findall(r"\[ema\] iter \d+  OA ([0-9.]+)  AA ([0-9.]+)", open(p).read())] or None
m5 = lambda v: None if not v else sum(v[-5:]) / len(v[-5:])
f = lambda v: "-" if v is None else f"{v:.2f}"
def table(title, group, tasks, key, ref):
    print(f"\n## {title}（{'OA' if 'OA' in key else 'AA'} 后 5 次均值；EMA 学生，括号内为原学生）\n")
    print("| 任务 | " + " | ".join(f"s{s}" for s in SEEDS) + " | 均值 ± 标准差（已完成的种子） |")
    print("|---|" + "---|" * (len(SEEDS) + 1))
    per_seed = {s: [] for s in SEEDS}
    for t in tasks:
        cells, vals = [], []
        for s in SEEDS:
            d = f"{group}/{t}_s{s}"
            e, r = m5(ema(d, "OA" if "OA" in key else "AA")), m5(detail(d, key))
            cells.append("-" if e is None else f"{e:.2f}（{f(r)}）")
            if e is not None:
                vals.append(e); per_seed[s].append(e)
        ms = "-" if not vals else (f"{st.mean(vals):.2f}" + (f" ± {st.stdev(vals):.2f}" if len(vals) > 1 else "") + f"（{len(vals)} 个）")
        print(f"| {t} | " + " | ".join(cells) + f" | {ms} |")
    full = [s for s in SEEDS if len(per_seed[s]) == len(tasks)]
    avgs = [st.mean(per_seed[s]) for s in full]
    if avgs:
        print(f"\n任务平均（只算 {len(tasks)} 个任务全部完成的种子）：" + "，".join(f"s{s} {a:.2f}" for s, a in zip(full, avgs))
              + (f"；均值 {st.mean(avgs):.2f} ± {st.stdev(avgs):.2f}" if len(avgs) > 1 else "") + f"（{ref}）")
OA, AA = "Overall Accuracy \\(OA\\)", "Average Accuracy \\(AA\\)"
table("Office-Home", os.environ["OH_G"], ["Ar2Cl", "Ar2Pr", "Ar2Re", "Cl2Ar", "Cl2Pr", "Cl2Re", "Pr2Ar", "Pr2Cl", "Pr2Re", "Re2Ar", "Re2Cl", "Re2Pr"], OA, "COSMO 86.41")
table("VisDA-C", os.environ["VD_G"], ["S2R"], AA, "COSMO 91.4")
table("DomainNet-126", os.environ["DN_G"], ["R2S", "R2C", "P2S", "C2S", "S2C", "P2C", "R2P", "C2P", "S2P", "P2R", "C2R", "S2R"], OA, "COSMO 85.1")
PY
echo "============================== 汇总结束 =============================="
