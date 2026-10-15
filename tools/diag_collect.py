"""
汇总 diag_ot.py 的结果，并打包成 zip 方便下载

用法：
  python tools/diag_collect.py --results runs/diag_results/2026_10_14 --process runs/process/Diag

输出：
  <results>/summary_all.csv   每个 (任务, 阶段) 一行的关键指标
  <results>/summary_all.md    同上，Markdown 表格，方便直接贴出来看
  <results>.zip               整个结果目录 + 训练日志
"""
import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、runs/process/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
import csv
import glob
import json
import shutil
import argparse


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.4f}"   # 边际误差这类很小的数用科学计数法
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="diag_ot.py 的结果根目录")
    parser.add_argument("--process", type=str, default="runs/process/Diag", help="训练输出目录（打包其中的日志）")
    a = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.results, "*", "*", "summary.json"))):
        s = json.load(open(path, encoding="utf-8"))
        st, cr = s["stats"], s["criteria"]
        rows.append({
            "任务": s["task"],
            "进度": s["progress"],
            "批数": s["n_batches"],
            "批内类数(均值)": get(st, "n_present", "mean"),
            "CLIP准确率": get(cr, "sanity", "clip_acc_batch"),
            "模型准确率": get(cr, "sanity", "model_acc_batch"),
            "计划argmax准确率": get(cr, "sanity", "plan_acc_bank_pre"),
            "行边际误差": get(cr, "sanity", "bank_row_err_mean"),
            "H1缺席质量_均匀μ": get(cr, "H1_absent_mass", "uniform_pre"),
            "H1缺席质量_实际μ": get(cr, "H1_absent_mass", "bank_pre"),
            "H1缺席质量_重加权后": get(cr, "H1_absent_mass", "bank_post(进入损失的计划)"),
            "H1弱(重加权后)": get(cr, "H1_absent_mass", "weak_bank_post"),
            "H2尺度翻倍最大变化pp": get(cr, "H2_scale", "joint_max_delta_pp_scale"),
            "H2ε翻倍最大变化pp": get(cr, "H2_scale", "joint_max_delta_pp_eps"),
            "H2弱": get(cr, "H2_scale", "weak_joint"),
            "H3代价比_Cv": get(cr, "H3_gap", "ratio_cv"),
            "H3代价比_Cv重加权": get(cr, "H3_gap", "ratio_cv_rw(重加权后)"),
            "H3代价比_Cm熵解": get(cr, "H3_gap", "ratio_cm_entropic(参考：Cm 自身熵正则解)"),
            "H3弱": get(cr, "H3_gap", "weak_cv"),
        })

    if not rows:
        print(f"[错误] {a.results} 下没有找到 summary.json"); return

    with open(os.path.join(a.results, "summary_all.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    keys = list(rows[0].keys())
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    with open(os.path.join(a.results, "summary_all.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # 复制训练日志（不含 .pt 快照，太大）
    if os.path.isdir(a.process):
        dst = os.path.join(a.results, "train_logs")
        for task_dir in sorted(glob.glob(os.path.join(a.process, "*"))):
            for name in ("argsInfo.txt", "run_log.txt", "ModellineAcc.txt"):
                src = os.path.join(task_dir, name)
                if os.path.isfile(src):
                    os.makedirs(os.path.join(dst, os.path.basename(task_dir)), exist_ok=True)
                    shutil.copy(src, os.path.join(dst, os.path.basename(task_dir), name))

    zip_path = shutil.make_archive(a.results.rstrip("/"), "zip", root_dir=a.results)
    print(f"\n已打包: {zip_path}（{os.path.getsize(zip_path) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
