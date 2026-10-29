"""
类边际诊断（读 rich.npz，不训练；标签只用于评估）：学生终点的错误里有多少来自「每类张数」估计错
对学生最终概率做 Sinkhorn 重分配到不同目标边际（无标签候选 vs 真实边际上界）：
  own  学生自身预测边际（对照，应约等于不变）  unif 均匀  tch 老师预测张数  mix 学生/老师张数几何平均  true 真实边际（上界，不可用）
用法：python tools/diag_prior.py runs/GateRich/cap02_tstrong/{Ar2Cl,Cl2Ar,Pr2Cl}_s42
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
import os
import numpy as np


def sinkhorn(P, r, it=200, lam=1.0):
    Q = P ** lam + 1e-12
    for _ in range(it):
        Q *= (r / Q.sum(0))[None]
        Q /= Q.sum(1, keepdims=True)
    return Q


for d in _sys.argv[1:]:
    z = np.load(os.path.join(d, "rich.npz"))
    y = z["label"]
    P, Tp = z["s_prob"][-1].astype(np.float64), z["t_prob"][-1].astype(np.float64)
    N, K = P.shape
    s, t = P.argmax(1), Tp.argmax(1)
    cs, ct, cy = [np.bincount(v, minlength=K) + 1.0 for v in (s, t, y)]
    tv = 0.5 * np.abs(cs / cs.sum() - cy / cy.sum()).sum()
    tvt = 0.5 * np.abs(ct / ct.sum() - cy / cy.sum()).sum()
    print(f"\n### {os.path.basename(d.rstrip('/'))}：学生 {np.mean(s == y) * 100:.2f}，老师 {np.mean(t == y) * 100:.2f}；预测边际与真实的 TV 距离 学生 {tv:.3f} 老师 {tvt:.3f}")
    for name, r in [("own", cs), ("unif", np.ones(K)), ("tch", ct), ("mix", np.sqrt(cs * ct)), ("true", cy)]:
        r = r / r.sum() * N
        for lam in (1.0, 0.5):
            q = sinkhorn(P, r, lam=lam).argmax(1)
            print(f"  {name:4s} λ={lam}：{np.mean(q == y) * 100:.2f}（{(np.mean(q == y) - np.mean(s == y)) * 100:+.2f}）")
    Pe = np.sqrt(P * Tp); Pe /= Pe.sum(1, keepdims=True)
    for name, r in [("ens", np.ones(K)), ("ens+true", cy)]:
        r = r / r.sum() * N
        q = sinkhorn(Pe, r).argmax(1) if name != "ens" else Pe.argmax(1)
        print(f"  师生几何平均 {name}：{np.mean(q == y) * 100:.2f}")
