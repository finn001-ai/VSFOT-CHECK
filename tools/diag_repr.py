"""
表示诊断：后期准确率停滞时，学生的表示有没有一起被老师同化；超出老师的信息在哪一层、用什么距离能拿到
（不训练，读取 main.py --snapshot_at 保存的快照）

背景（archive/2026_10_17_collapse）：师生收敛到共识不动点，分类器被 CLIP 同化；但 256 维瓶颈特征的簇纯度（K=520）
有 82.5%，高于并集。k-means 纯度依赖 K 和簇形状，这里换成不需要拆分的指标，并比较不同层：
  layer3      ResNet layer3 输出全局平均池化（1024 维），离对齐损失最远
  pool2048    骨干输出（2048 维，瓶颈之前；骨干学习率是瓶颈的 0.1 倍）
  bottleneck  瓶颈特征（256 维，分类器的输入，main.py 近邻平滑用的就是它）
  clip        老师（带 adapter 的 CLIP）的图像特征（512 维），参考
每层特征先减去全域均值再 L2 归一化，距离用余弦。

指标（标签只用于评估）：
  有效秩       中心化特征奇异值分布的 exp(熵)；90% 方差所需维数。后期下降 = 表示塌缩
  线性探针     5 折交叉验证的多类逻辑回归（特征 z-score，权重衰减在 3 个值里取最好）：表示里线性可分的信息上限
  近邻标签     k 个近邻真实标签的多数票（不含自身）：局部结构的好坏，不需要指定簇数
               knn = 普通 top-k；mutual = 只用互为近邻的（没有互近邻时退回普通）；csls = 去枢纽的 CSLS 距离
  无标签信号   方法能直接用的：(自身概率 + 近邻平均概率) / 2 的 argmax
               knn_s / mknn_s 用学生概率；knn_t / mknn_t 用老师概率（main.py 的 knn 平滑相当于 bottleneck 上的 knn_t）
               报告准确率、相对学生的差、救回的「两边都错」样本数、改坏的「学生对」样本数
  子集         学生判错的样本；锁死的样本（学生、老师都错且预测相同）上的探针 / 近邻标签准确率

用法（一般由 scripts/run_repr.sh 调用）：
  python tools/diag_repr.py --snapshot runs/ReprDiag/both_snap/Ar2Cl_s42/snapshot_p050.pt --out runs/repr_results/xxx/p050
  python tools/diag_repr.py --collect runs/repr_results/xxx --run_log runs/ReprDiag/both_snap/Ar2Cl_s42/run_log.txt
汇总写到 <collect>/summary.md 并完整打印（贴这一段即可）
"""
import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import glob
import json
import time
import argparse
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from diag_ot import DiagDataset
from utils.adapterVIT import clipAdapter
from model.network import resnet, classifier

SCALE = 100.0                                   # main.py 中写死的 CLIP 尺度
LAYERS = ["layer3", "pool2048", "bottleneck", "clip"]
SIGNALS = ["knn_s", "mknn_s", "knn_t", "mknn_t"]
K = 10                                          # 近邻数，和训练的 --knn_k 一致


# =============================================================================
# 近邻
# =============================================================================
def topk_neighbors(x, k, csls=False, chunk=1024):
    """x: [N, d] 已归一化；返回每个样本的 k 个近邻下标 [N, k]（不含自身）。csls：按 cos − r(y)/2 排序（r = 到自身 k 近邻的平均相似度）"""
    N = x.shape[0]
    r = None
    if csls:
        r = torch.zeros(N, device=x.device)
        for s in range(0, N, chunk):
            sim = x[s:s + chunk] @ x.T
            sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -2.0
            r[s:s + chunk] = sim.topk(k, dim=1).values.mean(1)
    out = torch.zeros(N, k, dtype=torch.long, device=x.device)
    for s in range(0, N, chunk):
        sim = x[s:s + chunk] @ x.T
        if csls:
            sim = sim - r.unsqueeze(0) / 2
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -1e9
        out[s:s + chunk] = sim.topk(k, dim=1).indices
    return out


def mutual_weights(nbr):
    """nbr: [N, k]；返回 [N, k] 的 0/1 权重（j 在 i 的近邻里且 i 也在 j 的近邻里），没有互近邻的行退回全 1；以及有互近邻的比例"""
    N = nbr.shape[0]
    A = torch.zeros(N, N, dtype=torch.bool, device=nbr.device)
    A[torch.arange(N, device=nbr.device).unsqueeze(1), nbr] = True
    w = A.T[torch.arange(N, device=nbr.device).unsqueeze(1), nbr].float()      # A[j, i]
    has = w.sum(1) > 0
    w[~has] = 1.0
    return w, has.float().mean().item()


def neighbor_mean(probs, nbr, w=None):
    """近邻概率的（加权）平均 [N, C]"""
    p = probs[nbr]                                                  # [N, k, C]
    if w is None:
        return p.mean(1)
    return (p * w.unsqueeze(2)).sum(1) / w.sum(1, keepdim=True)


def label_vote(y, nbr, C, w=None):
    """近邻真实标签的（加权）多数票"""
    oh = F.one_hot(y[nbr], C).float()                               # [N, k, C]
    if w is not None:
        oh = oh * w.unsqueeze(2)
    return oh.sum(1).argmax(1)


# =============================================================================
# 表示的指标
# =============================================================================
def effective_rank(x):
    """x: [N, d] 已中心化；返回 (exp(奇异值分布的熵), 90% 方差所需维数)"""
    s = torch.linalg.svdvals(x.double())
    p = s / s.sum()
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()).item()
    var = (s ** 2).cumsum(0) / (s ** 2).sum()
    return erank, int((var < 0.9).sum().item()) + 1


def linear_probe(x, y, C, folds=5, wds=(1e-4, 1e-3, 1e-2), seed=0):
    """5 折交叉验证的多类逻辑回归（LBFGS，全批）；返回 (最好的权重衰减, 每个样本的交叉验证预测)"""
    N = x.shape[0]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    g = torch.Generator().manual_seed(seed)
    fold = torch.randperm(N, generator=g).to(x.device) % folds
    best = (None, -1.0, None)
    for wd in wds:
        pred = torch.zeros(N, dtype=torch.long, device=x.device)
        for f in range(folds):
            tr, te = fold != f, fold == f
            W = torch.zeros(x.shape[1], C, device=x.device, requires_grad=True)
            b = torch.zeros(C, device=x.device, requires_grad=True)
            opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(x[tr] @ W + b, y[tr]) + wd * (W ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                pred[te] = (x[te] @ W + b).argmax(1)
        acc = (pred == y).float().mean().item()
        if acc > best[1]:
            best = (wd, acc, pred.clone())
    return best[0], best[2]


def sub_acc(pred, y, mask):
    return (pred[mask] == y[mask]).float().mean().item() if mask.any() else float("nan")


# =============================================================================
# 单个快照
# =============================================================================
def extract(snapshot, device, num_workers=4):
    """读取快照，全数据集一次前向（BN eval、测试变换）：返回 (快照, {层: 特征 [N, d]}, 学生概率, 老师概率, 标签)，按记忆库下标排序"""
    snap = torch.load(snapshot, map_location="cpu", weights_only=False)
    A = Namespace(**snap["args"])
    C = A.class_num
    print(f"[快照] {snapshot}  progress={snap['progress']}  iter={snap['iter']}/{snap['max_iter']}")

    raw_clip, preprocess = clip.load(A.clipVersion, device=device)
    clip_model = clipAdapter(raw_clip, device, adapterCount=A.adaptercount)
    missing, unexpected = clip_model.load_state_dict(snap["adapter"], strict=False)
    assert not unexpected and not [k for k in missing if "adapter" in k], "adapter 参数载入不完整"
    clip_model.eval()
    feature_net = resnet(res_name=A.arch).to(device)
    feature_net.load_state_dict(snap["feature"])
    feature_net.eval()
    classifier_net = classifier(class_num=C).to(device)
    classifier_net.load_state_dict(snap["classifier"])
    classifier_net.eval()

    dataset = DiagDataset(snap["imgs"], preprocess, A.class_info, isTraining=False)
    N = len(dataset)
    bb = feature_net.backbone                         # conv1, bn1, relu, maxpool, layer1..4, avgpool
    feats = {k: [] for k in LAYERS}
    idxs, q_m, q_v, ys = [], [], [], []
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in A.class_info]).to(device)
        tf = clip_model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for item in tqdm(DataLoader(dataset, batch_size=128, num_workers=num_workers), desc="前向"):
            h = bb[:7](item["img"].to(device).float())                 # 到 layer3
            feats["layer3"].append(h.mean(dim=(2, 3)))
            p = bb[7:](h).flatten(1)                                    # layer4 + avgpool
            feats["pool2048"].append(p)
            fb = feature_net.feat_bottleneck(p)
            feats["bottleneck"].append(fb)
            q_m.append(torch.softmax(classifier_net(fb), dim=1))
            gi = clip_model.encode_image(item["clip_img"].to(device).float())
            gi = gi / gi.norm(dim=-1, keepdim=True)
            feats["clip"].append(gi.float())
            q_v.append(torch.softmax(SCALE * gi @ tf.T, dim=1).float())
            idxs.append(item["index"].to(device)); ys.append(item["label"].to(device))
    order = torch.cat(idxs).argsort()
    feats = {k: torch.cat(v)[order].float() for k, v in feats.items()}
    q_m, q_v, y = torch.cat(q_m)[order], torch.cat(q_v)[order], torch.cat(ys)[order]
    return snap, feats, q_m, q_v, y


def run(cli):
    os.makedirs(cli.out, exist_ok=True)
    device = f"cuda:{cli.device}"
    t0 = time.time()
    snap, feats, q_m, q_v, y = extract(cli.snapshot, device, cli.num_workers)
    C, N = q_m.shape[1], q_m.shape[0]

    s_pred, t_pred = q_m.argmax(1), q_v.argmax(1)
    s_ok, t_ok = s_pred == y, t_pred == y
    s_wrong = ~s_ok
    locked = ~s_ok & ~t_ok & (s_pred == t_pred)
    res = {"progress": snap["progress"], "iter": snap["iter"], "N": N,
           "student": s_ok.float().mean().item(), "teacher": t_ok.float().mean().item(),
           "agree": (s_pred == t_pred).float().mean().item(), "union": (s_ok | t_ok).float().mean().item(),
           "both_wrong": int((~s_ok & ~t_ok).sum()), "locked": int(locked.sum()), "n_s_wrong": int(s_wrong.sum()),
           "layers": {}}
    print(f"[参考] 学生={res['student']:.4f}  老师={res['teacher']:.4f}  一致率={res['agree']:.4f}  并集={res['union']:.4f}  "
          f"两边都错={res['both_wrong']}  锁死={res['locked']}")

    for name in LAYERS:
        x = feats[name] - feats[name].mean(0)
        erank, d90 = effective_rank(x)
        xn = F.normalize(x, dim=1)
        r = {"dim": x.shape[1], "erank": erank, "d90": d90}
        with torch.enable_grad():
            wd, probe = linear_probe(x, y, C)
        r.update({"probe": (probe == y).float().mean().item(), "probe_wd": wd,
                  "probe_s_wrong": sub_acc(probe, y, s_wrong), "probe_locked": sub_acc(probe, y, locked)})
        nbr = topk_neighbors(xn, K)
        w, cover = mutual_weights(nbr)
        nbr_c = topk_neighbors(xn, K, csls=True)
        r["mutual_cover"] = cover
        for kind, pred in (("knn", label_vote(y, nbr, C)), ("mutual", label_vote(y, nbr, C, w)), ("csls", label_vote(y, nbr_c, C))):
            r[f"lab_{kind}"] = (pred == y).float().mean().item()
            r[f"lab_{kind}_s_wrong"] = sub_acc(pred, y, s_wrong)
            r[f"lab_{kind}_locked"] = sub_acc(pred, y, locked)
        for sig, probs, ww in (("knn_s", q_m, None), ("mknn_s", q_m, w), ("knn_t", q_v, None), ("mknn_t", q_v, w)):
            pred = ((probs + neighbor_mean(probs, nbr, ww)) / 2).argmax(1)
            ok = pred == y
            r[sig] = {"acc": ok.float().mean().item(), "d_student": ok.float().mean().item() - res["student"],
                      "saved": int((ok & ~s_ok & ~t_ok).sum()), "broke": int((~ok & s_ok).sum()), "fixed": int((ok & ~s_ok).sum())}
        res["layers"][name] = r
        print(f"[{name}] dim={r['dim']}  有效秩={erank:.1f}  90%方差维数={d90}  探针={r['probe']:.4f}（wd={wd}）  "
              f"近邻标签 knn/mutual/csls={r['lab_knn']:.4f}/{r['lab_mutual']:.4f}/{r['lab_csls']:.4f}  "
              + "  ".join(f"{s}={r[s]['acc']:.4f}" for s in SIGNALS))

    with open(os.path.join(cli.out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1)
    print(f"[完成] 用时 {time.time() - t0:.0f} 秒")


# =============================================================================
# 汇总：所有内容放进一个 summary.md 并打印
# =============================================================================
def collect(cli):
    files = sorted(glob.glob(os.path.join(cli.collect, "*", "result.json")))
    if not files:
        print(f"[错误] {cli.collect} 下没有 result.json"); return
    R = [json.load(open(f, encoding="utf-8")) for f in files]
    R.sort(key=lambda r: r["iter"])
    head = "| | " + " | ".join(f"p{round(r['progress'] * 100):03d}（{r['iter']}）" for r in R) + " |"
    sep = "|---|" + "---|" * len(R)
    p2 = lambda v: "-" if v != v else f"{v * 100:.2f}"

    md = ["# 表示诊断汇总", ""]
    # 训练日志：准确率曲线和耗时
    if cli.run_log and os.path.isfile(cli.run_log):
        pts, el = [], "-"
        for line in open(cli.run_log, encoding="utf-8"):
            m = re.search(r"Iter:\s*(\d+)\s*/\s*\d+; Now Acc: ([0-9.]+)", line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
            e = re.search(r"【总耗时】(\S+)", line)
            if e:
                el = e.group(1)
        if pts:
            md += [f"训练：最后 5 次评估均值 {p2(np.mean([a for _, a in pts[-5:]]))}，最终 {p2(pts[-1][1])}，"
                   f"最佳 {p2(max(a for _, a in pts))}，耗时 {el}", "",
                   "曲线（iter:acc）：" + "  ".join(f"{i}:{a * 100:.2f}" for i, a in pts[::4] + pts[-1:]), ""]

    md += ["## 1. 学生 / 老师（全域）", "", head, sep]
    for key, lab in (("student", "学生"), ("teacher", "老师"), ("agree", "一致率"), ("union", "并集")):
        md.append(f"| {lab} | " + " | ".join(p2(r[key]) for r in R) + " |")
    for key, lab in (("both_wrong", "两边都错"), ("locked", "锁死（都错且预测相同）"), ("n_s_wrong", "学生判错")):
        md.append(f"| {lab} | " + " | ".join(str(r[key]) for r in R) + " |")

    def layer_table(title, fn, note=None):
        rows = ["", f"## {title}", ""] + ([note, ""] if note else []) + [head.replace("| |", "| 层 |", 1), sep]
        for L in LAYERS:
            rows.append(f"| {L} | " + " | ".join(fn(r["layers"][L]) if L in r["layers"] else "-" for r in R) + " |")
        return rows

    md += layer_table("2. 有效秩 / 90% 方差维数", lambda l: f"{l['erank']:.1f} / {l['d90']}（{l['dim']}维）",
                      "下降 = 表示塌缩（信息被压进更少的方向）。")
    md += layer_table("3. 线性探针（5 折 CV，用标签，上限参考）", lambda l: p2(l["probe"]),
                      "和第 1 节的学生准确率比：高出的部分 = 表示里有、分类器没用上的信息。")
    md += layer_table("3b. 线性探针：学生判错 / 锁死样本上的准确率", lambda l: f"{p2(l['probe_s_wrong'])} / {p2(l['probe_locked'])}")
    for kind, lab in (("knn", "普通 kNN"), ("mutual", "互近邻"), ("csls", "CSLS")):
        md += layer_table(f"4. 近邻标签准确率（k={K}，{lab}）：全域 / 学生判错 / 锁死",
                          lambda l, kind=kind: f"{p2(l[f'lab_{kind}'])} / {p2(l[f'lab_{kind}_s_wrong'])} / {p2(l[f'lab_{kind}_locked'])}",
                          "互近邻覆盖率（有至少一个互近邻的样本比例）见第 4b 节。" if kind == "mutual" else None)
    md += layer_table("4b. 互近邻覆盖率", lambda l: p2(l["mutual_cover"]))
    for sig in SIGNALS:
        md += layer_table(f"5. 无标签信号 {sig}：准确率（相对学生） 救回两边都错 / 改坏学生对",
                          lambda l, sig=sig: f"{p2(l[sig]['acc'])}（{l[sig]['d_student'] * 100:+.2f}） {l[sig]['saved']}/{l[sig]['broke']}",
                          "knn_* = 普通近邻，mknn_* = 互近邻；*_s = 学生概率，*_t = 老师概率。" if sig == "knn_s" else None)
    text = "\n".join(md) + "\n"
    with open(os.path.join(cli.collect, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write(text)
    print("=" * 30 + " 汇总开始（把这一段完整贴回）" + "=" * 30)
    print(text)
    print("=" * 30 + " 汇总结束 " + "=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, help="main.py 保存的 snapshot_pXXX.pt")
    parser.add_argument("--out", type=str, help="结果输出目录")
    parser.add_argument("--collect", type=str, default=None, help="只做汇总：结果根目录")
    parser.add_argument("--run_log", type=str, default=None, help="汇总时附上训练曲线：训练目录下的 run_log.txt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    cli = parser.parse_args()
    if cli.collect:
        collect(cli)
    else:
        run(cli)
