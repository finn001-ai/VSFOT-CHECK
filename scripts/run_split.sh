#!/usr/bin/env bash
# =============================================================================
# 混淆类拆分诊断：训练早期快照（2026-10-22），Ar→Pr
#   p100 快照上（tools/diag_split.py）子簇可分、按真实标签命名上限 +0.6 ~ +1.5，但无标签命名全部失败——训练结束时概率饱和，
#   第二名信息被抹掉。「任一判对」在进度 5 ~ 10% 到 93.5 ~ 94，那时概率还没饱和：看早期快照上无标签命名是否可行。
#   底座同 base_sgd（cap02 + adapter + 标准提示上下文 SGD），40 epoch 调度，在进度 0.05 / 0.1 / 0.25 存快照，0.25 停止（约 13 分钟）。
#
#   !bash scripts/run_split.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="SplitSnap/base_sgd/Ar2Pr_s42"
OUT="runs/split_results/early"
if [ ! -f "runs/$NAME/.done" ]; then
    [ -d "runs/$NAME" ] && { echo "[错误] runs/$NAME 已存在但没有完成标记，请删除后重跑"; exit 1; }
    python -u main.py --device 0 --name "$NAME" --epochs 40 --dataset OfficeHomeDataset --source art --target product --seed 42 \
        --align_mode ot --knn_k 10 --align_view both --align_weight cap --align_cap 0.2 --t_prompt --t_prompt_opt sgd \
        --snapshot_at 0.05 0.1 0.25 --stop_at 0.25
    touch "runs/$NAME/.done"
fi
for p in 005 010 025; do
    python -u tools/diag_split.py --snapshot "runs/$NAME/snapshot_p$p.pt" \
        --zs_snapshot runs/H1Probe/p000/Ar2Pr_s42/snapshot_p000.pt --out "$OUT/p$p" 2>&1 | grep -v "%|" > "$OUT.p$p.log" || true
done
for p in 005 010 025; do
    echo "######## p$p"
    sed -n '/汇总开始/,/汇总结束/p' "$OUT.p$p.log"
done
