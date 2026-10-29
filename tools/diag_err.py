"""
最终错误构成诊断（读 samples.npz / rich_feat_*.npz，不训练；标签只用于评估）
问题：学生终点 ≈ 训练前师生并集 + 0.5，剩下的错误是什么？
  1) 最终错误中：训练前两边都错 / 只有一边对 / 两边都对（后来学错）各占多少
  2) 最终错误时师生是否给出同一个错标签（共享错误 = 互学锁死）
  3) 最终错误集中在哪些 (真类 → 预测类) 对上
  4) 最后一个有特征的评估点上，在学生 / 老师特征里做 kNN 多数投票（用学生预测作为投票标签），能救回多少错误、新增多少错误
用法：python tools/diag_err.py runs/GateRich/cap02_tstrong/{Ar2Cl,Cl2Ar,Pr2Cl}_s42
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
import os, glob
import numpy as np
import torch
import torch.nn.functional as F

NAMES = [l.strip() for l in open("dataset/OfficeHomeDataset/classname.txt", encoding="utf-8") if l.strip()]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def knn_vote(feat, lab, K, k):
    X = F.normalize(torch.tensor(feat, dtype=torch.float32, device=DEV), dim=1)
    y = torch.tensor(lab, device=DEV)
    sim = X @ X.T
    w, nb = sim.topk(k + 1, dim=1)
    w, nb = w[:, 1:], nb[:, 1:]
    votes = torch.zeros(len(y), K, device=DEV).scatter_add_(1, y[nb], torch.exp(w / 0.1))
    return votes.argmax(1).cpu().numpy()


def analyse(d):
    z = np.load(os.path.join(d, "samples.npz"))
    y, S, T = z["label"], z["s_pred"], z["t_pred"]
    K = len(NAMES)
    s0, t0, sf, tf = S[0], T[0], S[-1], T[-1]
    print(f"\n### {os.path.basename(d.rstrip('/'))}（N={len(y)}）")
    print(f"训练前 学生 {np.mean(s0 == y) * 100:.2f} 老师 {np.mean(t0 == y) * 100:.2f} 并集 {np.mean((s0 == y) | (t0 == y)) * 100:.2f}；终点 学生 {np.mean(sf == y) * 100:.2f} 老师 {np.mean(tf == y) * 100:.2f} 并集 {np.mean((sf == y) | (tf == y)) * 100:.2f}")
    err = sf != y
    both0 = (s0 != y) & (t0 != y)
    one0 = (s0 == y) ^ (t0 == y)
    print(f"最终错误 {err.sum()} 张：训练前两边都错 {np.sum(err & both0)}，只一边对 {np.sum(err & one0)}，两边都对 {np.sum(err & (s0 == y) & (t0 == y))}")
    print(f"训练前两边都错 {both0.sum()} 张，终点学生救回 {np.sum(both0 & ~err)}")
    print(f"最终错误中师生同一错标签 {np.sum(err & (sf == tf))}，老师对 {np.sum(err & (tf == y))}，各错各的 {np.sum(err & (tf != y) & (tf != sf))}")
    pairs = {}
    for a, b in zip(y[err], sf[err]):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    top = sorted(pairs.items(), key=lambda x: -x[1])[:12]
    cum = sum(v for _, v in top)
    print(f"前 12 个混淆对占错误 {cum / err.sum() * 100:.1f}%：" + "；".join(f"{NAMES[a]}→{NAMES[b]} {v}" for (a, b), v in top))
    # 类内错误集中度
    per = np.bincount(y[err], minlength=K) / np.maximum(np.bincount(y, minlength=K), 1)
    worst = np.argsort(-per)[:8]
    print("错误率最高的类：" + "；".join(f"{NAMES[k]} {per[k] * 100:.0f}%" for k in worst))
    feats = sorted(glob.glob(os.path.join(d, "rich_feat_*.npz")))
    if feats:
        f = np.load(feats[-1])
        it = int(os.path.basename(feats[-1])[10:15])
        r = np.load(os.path.join(d, "rich.npz"))
        e = int(np.where(r["iter"] == it)[0][0])
        s_e = r["s_prob"][e].argmax(1)
        t_e = r["t_prob"][e].argmax(1)
        print(f"特征点 iter {it}：学生 {np.mean(s_e == y) * 100:.2f} 老师 {np.mean(t_e == y) * 100:.2f}")
        for name in ("s_feat", "t_feat"):
            for k in (5, 10, 20, 50):
                v = knn_vote(f[name], s_e, K, k)
                print(f"  {name} kNN{k} 投票（学生标签）：{np.mean(v == y) * 100:.2f}（救回 {np.sum((s_e != y) & (v == y))} / 新错 {np.sum((s_e == y) & (v != y))}）")
            v = knn_vote(f[name], t_e, K, 10)
            print(f"  {name} kNN10 投票（老师标签）：{np.mean(v == y) * 100:.2f}")
        cat = np.concatenate([F.normalize(torch.tensor(f["s_feat"], dtype=torch.float32), dim=1).numpy(), F.normalize(torch.tensor(f["t_feat"], dtype=torch.float32), dim=1).numpy()], 1)
        v = knn_vote(cat, s_e, K, 10)
        print(f"  拼接特征 kNN10 投票（学生标签）：{np.mean(v == y) * 100:.2f}")


if __name__ == "__main__":
    for d in _sys.argv[1:]:
        analyse(d)
