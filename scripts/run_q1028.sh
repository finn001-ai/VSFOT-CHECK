#!/usr/bin/env bash
# =============================================================================
# 2026-10-28 队列（seed 42），全部串行（一个 OH 任务占约 24G 显存，两组并行会 OOM）
#   第 1 步：Office-Home Cl→Ar 补析因的两格（40 epoch）
#     Cl→Ar 上「学生 > 老师」的评估点比例只有 0.17，接近 DN / VisDA（0.00 ~ 0.07），但从来没加过老师侧共识。
#     cap02_tstrong_ct      = cap02_tstrong + --consensus teacher                （inv + 共识）
#     cap02_tstrong_self_ct = cap02_tstrong + --clsw_mode self --consensus teacher（self + 共识）
#     对照（已有）：cap02_tstrong 83.77（inv，无共识）、cap02_tstrong_self 84.04（self，无共识）
#   第 2 步：VisDA-C 30% 子集 3 epoch，老师目标的尖锐度（不加共识）
#     MGVA 目标 = softmax(10 · 归一化 top-k)，候选外质量随 k 和类别数变；共识把 top-k 权重变得接近 one-hot。
#     q30_norm65_topk1 / topk2 = q30_norm65 + --topk 1 / 2，其余参数和 q30_norm65 完全相同
#     对照（已有）：q30_norm65 88.06（topk 3，无共识）、q30_norm65_cons_t 88.56（topk 3，老师侧共识）
#   第 3 步：VisDA-C 30% 子集，IM 的 diversity 改成正确符号
#     utils/Loss.py：loss = 样本熵 − imloss · Σ m log m = 样本熵 + imloss · H(批平均)，原版代码是在压低批平均熵（鼓励塌缩）。
#     --imloss -1 = 样本熵 − H(批平均)，即标准 IM，不改代码。q30_norm65_imfix，对照 q30_norm65
#
#   !bash scripts/run_q1028.sh            中途看汇总：!SUMMARY_ONLY=1 bash scripts/run_q1028.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."    # 切换到项目根目录（本脚本在 scripts/ 下）

PYTHON="python -u"
GPU=0
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

OH_BASE="--align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_strong"
VD_BASE="--align_mode ot --knn_k 10 --align_view both --arch resnet101 --workers 12 --align_weight cap --align_cap 0.2 --t_strong \
--epochs 3 --batch_size 32 --data_frac 0.3 --clsw_mode self --align_norm_c 65"

# 格式 "目录|数据集|源域|目标域|参数"
STEP1=(
    "WeightSched/cap02_tstrong_ct/Cl2Ar_s42|OfficeHomeDataset|clipart|art|--epochs 40 $OH_BASE --consensus teacher"
    "WeightSched/cap02_tstrong_self_ct/Cl2Ar_s42|OfficeHomeDataset|clipart|art|--epochs 40 $OH_BASE --clsw_mode self --consensus teacher"
)
STEP2=(
    "VisDA/q30_norm65_topk1/S2R_s42|VisDA|synthesis|real|$VD_BASE --topk 1"
    "VisDA/q30_norm65_topk2/S2R_s42|VisDA|synthesis|real|$VD_BASE --topk 2"
)
STEP3=(
    "VisDA/q30_norm65_imfix/S2R_s42|VisDA|synthesis|real|$VD_BASE --imloss -1"
)
mkdir -p runs/q1028_logs

run_one() {
    IFS="|" read -r dir ds s t extra <<< "$1"
    if [ -f "runs/$dir/.done" ]; then echo "==> $dir：已完成，跳过"; return 0; fi
    if [ -d "runs/$dir" ]; then echo "[错误] runs/$dir 已存在但没有完成标记（可能上次中断），请改名后重跑"; return 1; fi
    echo "==> 开始 $dir  参数：$extra"
    # $extra 不加引号，让它拆成多个参数；调用方带 || 时 set -e 不生效，所以显式检查退出码，失败不写完成标记
    if ! $PYTHON main.py --device "$GPU" --name "$dir" --dataset "$ds" --source "$s" --target "$t" --seed 42 $extra \
        > "runs/q1028_logs/$(echo "$dir" | tr / _).log" 2>&1; then
        echo "[错误] $dir 训练失败，见 runs/q1028_logs/"; return 1
    fi
    touch "runs/$dir/.done"
    echo "==> 完成 $dir"
}

run_step() {
    local rc=0
    for item in "$@"; do run_one "$item" || rc=1; done
    return $rc
}

if [ "$SUMMARY_ONLY" != "1" ]; then
    T0=$(date +%s)
    run_step "${STEP1[@]}" || echo "[警告] 第 1 步有任务失败，见 runs/q1028_logs/"
    T1=$(date +%s)
    run_step "${STEP2[@]}" || echo "[警告] 第 2 步有任务失败，见 runs/q1028_logs/"
    T2=$(date +%s)
    run_step "${STEP3[@]}" || echo "[警告] 第 3 步有任务失败，见 runs/q1028_logs/"
    T3=$(date +%s)
    echo "耗时：第 1 步 $(( (T1 - T0) / 60 )) 分钟，第 2 步 $(( (T2 - T1) / 60 )) 分钟，第 3 步 $(( (T3 - T2) / 60 )) 分钟"
fi

echo "============================== 汇总开始（把这一段完整贴回）=============================="
$PYTHON - <<'EOF'
import os, re
def evals(d, key):
    p = f"runs/{d}/Detail.txt"
    if not os.path.exists(p):
        return None
    v = [float(x) * 100 for x in re.findall(key + r": ([0-9.]+)", open(p).read())][1:]   # 去掉第 0 步
    return v or None
def st(d):
    return "完成" if os.path.exists(f"runs/{d}/.done") else ("进行中" if os.path.exists(f"runs/{d}") else "未开始")
f = lambda v: "-" if v is None else f"{v:.2f}"

print("## 第 1 步：Office-Home Cl→Ar（40 epoch，seed 42；OA，均值 = 最后 5 次评估）\n")
print("| 组 | 类均衡 | 老师侧共识 | OA 后 5 均值 | OA 最后一次 | 相对 cap02_tstrong | 状态 |")
print("|---|---|---|---|---|---|---|")
OA = "Overall Accuracy \\(OA\\)"
rows = [("cap02_tstrong", "inv", "无"), ("cap02_tstrong_self", "self", "无"),
        ("cap02_tstrong_ct", "inv", "有"), ("cap02_tstrong_self_ct", "self", "有")]
ref = evals("WeightSched/cap02_tstrong/Cl2Ar_s42", OA)
ref5 = sum(ref[-5:]) / len(ref[-5:]) if ref else None
for g, cw, ct in rows:
    d = f"WeightSched/{g}/Cl2Ar_s42"
    v = evals(d, OA)
    m = sum(v[-5:]) / len(v[-5:]) if v else None
    dd = "-" if None in (m, ref5) else f"{m - ref5:+.2f}"
    print(f"| {g} | {cw} | {ct} | {f(m)} | {f(v[-1] if v else None)} | {dd} | {st(d)} |")
print("\nCOSMO Cl→Ar 85.7；oracle 先验上界（cap02_tstrong_oracle）", end=" ")
v = evals("WeightSched/cap02_tstrong_oracle/Cl2Ar_s42", OA)
print(f(sum(v[-5:]) / len(v[-5:]) if v else None))

print("\n## 第 2 步：VisDA-C 30% 子集 3 epoch（norm65 + self + tstrong，seed 42；均值 = 不含第 0 步的全部评估）\n")
print("| 组 | topk | 老师侧共识 / 其他 | AA 均值 | AA 最后一次 | OA 均值 | 相对 q30_norm65（AA） | 状态 |")
print("|---|---|---|---|---|---|---|---|")
AA = "Average Accuracy \\(AA\\)"
rows = [("q30_norm65", 3, "无"), ("q30_norm65_topk2", 2, "无"), ("q30_norm65_topk1", 1, "无"), ("q30_norm65_cons_t", 3, "有"), ("q30_norm65_imfix", 3, "无；IM 正确符号")]
ref = evals("VisDA/q30_norm65/S2R_s42", AA)
refm = sum(ref) / len(ref) if ref else None
for g, k, ct in rows:
    d = f"VisDA/{g}/S2R_s42"
    a, o = evals(d, AA), evals(d, OA)
    am = sum(a) / len(a) if a else None
    dd = "-" if None in (am, refm) else f"{am - refm:+.2f}"
    print(f"| {g} | {k} | {ct} | {f(am)} | {f(a[-1] if a else None)} | {f(sum(o) / len(o) if o else None)} | {dd} | {st(d)} |")
print("\n各类准确率（最后一次评估）：")
for g, _, _ in rows:
    p = f"runs/VisDA/{g}/S2R_s42/ModellineAcc.txt"
    if os.path.exists(p):
        L = open(p).read().strip().splitlines()
        print(g); print(L[0]); print(L[-1])
EOF
echo "============================== 汇总结束 =============================="
