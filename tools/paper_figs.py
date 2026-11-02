"""
论文诊断图（不训练；标签只用于评估和作图）
  extract：从 final.pt（EMA 学生 + 老师 adapter）/ Source.pt 在目标域全部图片上算学生特征、学生概率、老师概率，存 runs/paper_figs/cache/<名字>.npz
  plot：读缓存和 runs/PaperDyn/*/dynamics.csv、samples.npz，画
    fig_tsne_*.pdf        学生特征 t-SNE：源模型 / VSFOT / 本文（同一批样本、同一 t-SNE 设置）
    fig_dyn_*.pdf         训练动态：学生准确率、对齐目标准确率（A）、师生同一错标签的比例（B）
    fig_visda_class.pdf   VisDA 各类准确率与预测类分布 vs 真实分布（C）
    fig_visda_cm.pdf      VisDA 混淆矩阵：VSFOT vs 本文
    numbers.md            图里对应的数字（kNN 准确率、轮廓系数、共同错误、预测分布 L1 等），写正文用
用法：python tools/paper_figs.py extract     python tools/paper_figs.py plot
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
_sys.path.insert(0, _ROOT)
import os, sys, glob, csv
import numpy as np
import torch
import torch.nn.functional as F

OUT = "runs/paper_figs"
CACHE = os.path.join(OUT, "cache")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# 名字 → (checkpoint, 类型)。source = 源模型（老师为零样本 CLIP）；final = 训练结束（汇报的 EMA 学生 + 训练后的老师）
# 「完整」行用 PaperDyn 重跑（seed 42 原目录没有 final.pt，重跑与原结果逐位一致，见 run_paper_dyn.sh 汇总）
RUNS = {
    "oh_source": ("runs/Source_home/Source_art/Source.pt", "source", "OfficeHomeDataset", "Clipart"),
    "oh_vsfot":  ("runs/PaperDyn/oh_vsfot/Ar2Cl_s42/final.pt", "final", "OfficeHomeDataset", "Clipart"),
    "oh_minusA": ("runs/PaperDyn/oh_minusA/Ar2Cl_s42/final.pt", "final", "OfficeHomeDataset", "Clipart"),
    "oh_minusB": ("runs/PaperDyn/oh_minusB/Ar2Cl_s42/final.pt", "final", "OfficeHomeDataset", "Clipart"),
    "oh_full":   ("runs/PaperDyn/oh_full/Ar2Cl_s42/final.pt", "final", "OfficeHomeDataset", "Clipart"),
    "vd_source": ("runs/Source_visda/Source_synthesis/Source.pt", "source", "VisDA", "real"),
    "vd_vsfot":  ("runs/PaperDyn/vd_vsfot/S2R_s42/final.pt", "final", "VisDA", "real"),
    "vd_minusA": ("runs/PaperDyn/vd_minusA/S2R_s42/final.pt", "final", "VisDA", "real"),
    "vd_minusB": ("runs/PaperDyn/vd_minusB/S2R_s42/final.pt", "final", "VisDA", "real"),
    "vd_minusC": ("runs/PaperDyn/vd_minusC/S2R_s42/final.pt", "final", "VisDA", "real"),
    "vd_full":   ("runs/PaperDyn/vd_full/S2R_s42/final.pt", "final", "VisDA", "real"),
    "dn_source": ("runs/Source_domainnet/Source_sketch/Source.pt", "source", "DomainNet", "painting"),
    "dn_full":   ("runs/DomainNet/cap02_tstrong_self_ct_sema_e10/S2P_s42/final.pt", "final", "DomainNet", "painting"),
}
ARCH = {"OfficeHomeDataset": "resnet50", "VisDA": "resnet101", "DomainNet": "resnet50"}
LABEL = {"source": "Source only", "vsfot": "VSFOT", "minusA": "w/o A", "minusB": "w/o B", "minusC": "w/o C", "full": "Ours"}

# 参考调色板（dataviz palette.md，亮色），按固定顺序分配：本文永远是 1 号色
SERIES = {"full": "#2a78d6", "minusA": "#eb6834", "minusB": "#1baf7a", "minusC": "#eda100", "vsfot": "#6250d6"}
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"


# ----------------------------------------------------------------------------- extract
def class_info(ds):
    from utils.Other import load_class_names
    if ds == "OfficeHomeDataset":
        return load_class_names("dataset/OfficeHomeDataset/classname.txt", 65)
    if ds == "VisDA":       # 同 main.py
        return ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    for p in glob.glob("runs/PaperDyn/*/*/final.pt") + glob.glob("runs/DomainNet/cap02_tstrong_self_ct_sema_e10/*_s42/final.pt"):
        a = torch.load(p, map_location="cpu", weights_only=False)["args"]
        if a["dataset"] == ds:
            return a["class_info"]
    raise RuntimeError(f"找不到 {ds} 的类名")


class EvalSet(torch.utils.data.Dataset):
    """目标域全部图片，固定顺序（列表文件顺序），只做测试变换（学生）和 CLIP 预处理（老师）"""
    def __init__(self, ds, domain, clip_pre, names):
        from utils.Dataset import dataset5X
        lines = open(os.path.join("dataset", ds, f"{domain}.txt")).read().split("\n")
        # 与 utils/Dataset.py getDataloaderSFDA5X 相同的路径拼接（OH 列表是绝对路径，join 后不变）
        self.items = [(os.path.join("dataset", ds, domain, l.split()[0]), int(l.split()[1])) for l in lines if l.strip()]
        ref = dataset5X(image_list=[f"{p},{y}" for p, y in self.items[:1]], clipProcess=clip_pre, classDetail=names, isTraining=False)
        self.tf, self.ctf = ref.test_transform, ref.clipProcessVal

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from utils.Dataset import RGB_loader
        p, y = self.items[i]
        img = RGB_loader(p)
        return self.tf(img), self.ctf(img), y


@torch.no_grad()
def extract():
    import clip
    from model.network import resnet, classifier
    from utils.adapterVIT import clipAdapter
    os.makedirs(CACHE, exist_ok=True)
    raw, pre = clip.load("ViT-B/32", device=DEV)
    loaders = {}
    for name, (ck, kind, ds, dom) in RUNS.items():
        out = os.path.join(CACHE, f"{name}.npz")
        if os.path.exists(out):
            print(f"{name}：已有缓存，跳过"); continue
        if not os.path.exists(ck):
            print(f"{name}：没有 {ck}，跳过"); continue
        names = class_info(ds)
        if (ds, dom) not in loaders:
            loaders[(ds, dom)] = torch.utils.data.DataLoader(EvalSet(ds, dom, pre, names), batch_size=128, shuffle=False, num_workers=12)
        z = torch.load(ck, map_location="cpu", weights_only=False)
        fnet, cnet = resnet(res_name=ARCH[ds]).to(DEV), classifier(class_num=len(names)).to(DEV)
        if kind == "source":
            fnet.load_state_dict(z["best_feature"]); cnet.load_state_dict(z["best_classifier"])
            teacher = raw.float()
        else:
            fnet.load_state_dict(z["ema_feature"]); cnet.load_state_dict(z["ema_classifier"])
            teacher = clipAdapter(raw, DEV, adapterCount=z["args"]["adaptercount"])
            miss = teacher.load_state_dict(z["adapter"], strict=False)
            assert not miss.unexpected_keys and all("adapter" not in k for k in miss.missing_keys), "adapter 权重没有全部载入"
        fnet.eval(); cnet.eval(); teacher.eval()
        tf = F.normalize(teacher.encode_text(clip.tokenize([f"a photo of a {c}" for c in names]).to(DEV)).float(), dim=-1)
        S, P, T, Y = [], [], [], []
        for x, xc, y in loaders[(ds, dom)]:
            f = fnet(x.to(DEV).float())
            S.append(F.normalize(f, dim=-1).half().cpu()); P.append(torch.softmax(cnet(f), 1).half().cpu())
            vf = F.normalize(teacher.encode_image(xc.to(DEV).float()).float(), dim=-1)
            T.append(torch.softmax(100 * vf @ tf.T, 1).half().cpu()); Y.append(y)
        S, P, T, Y = torch.cat(S).numpy(), torch.cat(P).numpy(), torch.cat(T).numpy(), torch.cat(Y).numpy()
        np.savez_compressed(out, feat=S, s_prob=P, t_prob=T, label=Y)
        pred = P.argmax(1)
        cm = np.bincount(Y * len(names) + pred, minlength=len(names) ** 2).reshape(len(names), -1)
        print(f"{name}：N={len(Y)}  学生 OA {np.mean(pred == Y) * 100:.2f}  AA {np.mean(cm.diagonal() / cm.sum(1)) * 100:.2f}  "
              f"老师 OA {np.mean(T.argmax(1) == Y) * 100:.2f}（final.pt 对应 EMA 学生，应与 ModellineAcc_ema.txt 最后一行一致）")
        if kind == "final":
            del teacher
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- plot
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    return plt


def load(name):
    p = os.path.join(CACHE, f"{name}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def knn_acc(feat, y, k=10):
    """特征空间 kNN（余弦，k=10，留一）分类准确率：特征的类可分性，与分类头无关"""
    X = torch.tensor(feat, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, device=DEV)
    ok = 0
    for i in range(0, len(X), 2048):
        sim = X[i:i + 2048] @ X.T
        sim[torch.arange(sim.size(0)), torch.arange(i, i + sim.size(0))] = -2
        nb = sim.topk(k, dim=1).indices
        ok += (torch.mode(yt[nb], dim=1).values == yt[i:i + 2048]).sum().item()
    return ok / len(X) * 100


def tsne(plt, prefix, rows, names, title, n_per_class, highlight, lines):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    data = [(r, load(f"{prefix}_{r}")) for r in rows]
    data = [(r, d) for r, d in data if d is not None]
    if len(data) < 2:
        print(f"t-SNE {prefix}：缓存不足，跳过"); return
    y = data[0][1]["label"]
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:n_per_class] for c in range(len(names))])
    hl = [c for c in highlight]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axes = plt.subplots(1, len(data), figsize=(2.6 * len(data), 2.7))
    lines.append(f"\n### t-SNE {title}（每类最多 {n_per_class} 张，共 {len(idx)} 张；kNN 与轮廓系数在全部样本 / 同一子集上算）\n")
    lines.append("| 模型 | 学生 OA | 特征 kNN 准确率（全部样本） | 轮廓系数（子集，余弦） |\n|---|---|---|---|")
    for ax, (r, d) in zip(np.atleast_1d(axes), data):
        assert np.array_equal(d["label"], y), "各模型的样本顺序不一致"
        emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", metric="cosine", random_state=0).fit_transform(d["feat"][idx].astype(np.float32))
        yy = y[idx]
        other = ~np.isin(yy, hl)
        ax.scatter(emb[other, 0], emb[other, 1], s=1.5, c=MUTED, linewidths=0, rasterized=True)
        for j, c in enumerate(hl):
            m = yy == c
            ax.scatter(emb[m, 0], emb[m, 1], s=3, c=pal[j], linewidths=0, rasterized=True, label=names[c])
        ax.set_title(LABEL[r], color=INK); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        oa = np.mean(d["s_prob"].argmax(1) == y) * 100
        sil = silhouette_score(d["feat"][idx].astype(np.float32), yy, metric="cosine")
        ka = knn_acc(d["feat"], y)
        ax.text(0.02, 0.02, f"kNN {ka:.1f}  sil {sil:.3f}", transform=ax.transAxes, fontsize=7, color=INK2)
        lines.append(f"| {LABEL[r]} | {oa:.2f} | {ka:.2f} | {sil:.3f} |")
    np.atleast_1d(axes)[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3, fontsize=7, labelcolor=INK2)
    fig.suptitle(title, color=INK, fontsize=9, y=1.02)
    fig.savefig(os.path.join(OUT, f"fig_tsne_{prefix}.pdf")); fig.savefig(os.path.join(OUT, f"fig_tsne_{prefix}.png"))
    plt.close(fig)


def read_dyn(d):
    p = os.path.join(d, "dynamics.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    out = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    z = np.load(os.path.join(d, "samples.npz"))
    y, S, T = z["label"], z["s_pred"], z["t_pred"]
    out["shared_err"] = ((S == T) & (S != y[None])).mean(1) * 100       # 师生给同一个错标签的样本比例
    out["shared_of_err"] = ((S == T) & (S != y[None])).sum(1) / np.maximum((S != y[None]).sum(1), 1) * 100
    out["prog"] = out["iter"] / out["iter"].max()
    return out


def dynamics(plt, task, rows, title, lines):
    data = [(r, read_dyn(f"runs/PaperDyn/{task}_{r}/" + ("Ar2Cl_s42" if task == "oh" else "S2R_s42"))) for r in rows]
    data = [(r, d) for r, d in data if d is not None]
    if not data:
        print(f"训练动态 {task}：没有数据，跳过"); return
    panels = [("student_acc", "Student accuracy (%)", 100), ("pl_align_acc", "Alignment-target accuracy (%)", 100),
              ("shared_err", "Same wrong label, student & teacher (%)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (k, lab, sc) in zip(axes, panels):
        for r, d in data:
            v = d[k] * sc
            m = np.isfinite(v)
            ax.plot(d["prog"][m], v[m], color=SERIES[r], label=LABEL[r])
            ax.annotate(f"{v[m][-1]:.1f}", (d["prog"][m][-1], v[m][-1]), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=INK2, va="center")
        ax.set_title(lab, color=INK); ax.set_xlabel("Training progress")
        ax.set_xlim(0, 1.12)
    axes[0].legend(fontsize=7, labelcolor=INK2, handlelength=1.5)
    fig.suptitle(title, color=INK, fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_dyn_{task}.pdf")); fig.savefig(os.path.join(OUT, f"fig_dyn_{task}.png"))
    plt.close(fig)
    lines.append(f"\n### 训练动态 {title}（终点；对齐目标准确率取最后一个 epoch 训练中的统计）\n")
    lines.append("| 模型 | 学生 OA | 老师 OA | 对齐目标准确率 | 计划 argmax 准确率 | 师生一致率 | 师生同一错标签（占全部 / 占学生错误） |\n|---|---|---|---|---|---|---|")
    for r, d in data:
        lines.append(f"| {LABEL[r]} | {d['student_acc'][-1] * 100:.2f} | {d['teacher_acc'][-1] * 100:.2f} | {d['pl_align_acc'][-1] * 100:.2f} | "
                     f"{d['pl_gamma_acc'][-1] * 100:.2f} | {d['agree'][-1] * 100:.2f} | {d['shared_err'][-1]:.2f} / {d['shared_of_err'][-1]:.1f} |")


def visda_class(plt, names, lines):
    rows = [r for r in ("vsfot", "minusC", "full") if load(f"vd_{r}") is not None]
    if not rows:
        print("VisDA 各类：没有缓存，跳过"); return
    D = {r: load(f"vd_{r}") for r in rows}
    y = D[rows[0]]["label"]
    C = len(names)
    true = np.bincount(y, minlength=C) / len(y) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    w = 0.8 / len(rows)
    x = np.arange(C)
    lines.append("\n### VisDA 各类准确率（EMA 学生）与预测类分布\n")
    lines.append("| 模型 | " + " | ".join(names) + " | AA | 预测分布与真实分布 L1 |\n|---|" + "---|" * (C + 2))
    for j, r in enumerate(rows):
        pred = D[r]["s_prob"].argmax(1)
        acc = np.array([np.mean(pred[y == c] == c) for c in range(C)]) * 100
        axes[0].bar(x + (j - (len(rows) - 1) / 2) * w, acc, width=w * 0.9, color=SERIES[r], label=LABEL[r])
        dist = np.bincount(pred, minlength=C) / len(pred) * 100
        if r != "vsfot":
            axes[1].plot(x, dist, color=SERIES[r], marker="o", markersize=3.5, label=LABEL[r])
        lines.append(f"| {LABEL[r]} | " + " | ".join(f"{a:.1f}" for a in acc) + f" | {acc.mean():.2f} | {np.abs(dist - true).sum():.2f} |")
    lines.append("| 真实类比例（%） | " + " | ".join(f"{t:.1f}" for t in true) + " | | |")
    axes[0].set_ylim(40, 100); axes[0].set_ylabel("Per-class accuracy (%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].legend(fontsize=7, labelcolor=INK2, ncol=len(rows), loc="lower left")
    axes[1].bar(x, true, width=0.7, color=GRID, label="Ground truth")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha="right"); axes[1].set_ylabel("Share of predictions (%)")
    axes[1].legend(fontsize=7, labelcolor=INK2)
    axes[0].set_title("VisDA-C per-class accuracy", color=INK); axes[1].set_title("Predicted vs. true class distribution", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_visda_class.pdf")); fig.savefig(os.path.join(OUT, "fig_visda_class.png"))
    plt.close(fig)


def visda_cm(plt, names):
    rows = [r for r in ("vsfot", "full") if load(f"vd_{r}") is not None]
    if len(rows) < 2:
        print("VisDA 混淆矩阵：缓存不足，跳过"); return
    C = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for ax, r in zip(axes, rows):
        d = load(f"vd_{r}")
        y, p = d["label"], d["s_prob"].argmax(1)
        cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C).astype(float)
        cm = cm / cm.sum(1, keepdims=True) * 100
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(C):
            for j in range(C):
                if cm[i, j] >= 1:
                    ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="white" if cm[i, j] > 60 else INK)
        ax.set_xticks(range(C)); ax.set_yticks(range(C))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.5); ax.set_yticklabels(names, fontsize=6.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
        ax.set_title(LABEL[r], color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Row-normalized (%)")
    fig.savefig(os.path.join(OUT, "fig_visda_cm.pdf")); fig.savefig(os.path.join(OUT, "fig_visda_cm.png"))
    plt.close(fig)


def plot():
    plt = style()
    os.makedirs(OUT, exist_ok=True)
    lines = ["# 论文诊断图对应的数字（seed 42；学生均为 EMA 学生，标签只用于评估）"]
    oh_names, vd_names = class_info("OfficeHomeDataset"), class_info("VisDA")
    # t-SNE 只给 8 个类上色（参考调色板的类别上限）：源模型在目标域上准确率最低的 8 个类（事先定的规则，不看适配后的结果），其余灰色
    src = load("oh_source")
    if src is not None:
        pred, y = src["s_prob"].argmax(1), src["label"]
        hard = np.argsort([np.mean(pred[y == c] == c) for c in range(65)])[:8]
        tsne(plt, "oh", ["source", "vsfot", "full"], oh_names, "Office-Home Ar→Cl", 1000, list(hard), lines)
    vs = load("vd_source")
    if vs is not None:
        pred, y = vs["s_prob"].argmax(1), vs["label"]
        hard = np.argsort([np.mean(pred[y == c] == c) for c in range(12)])[:8]
        tsne(plt, "vd", ["source", "vsfot", "full"], vd_names, "VisDA-C S→R", 300, list(hard), lines)
    dn = load("dn_source")
    if dn is not None:
        dn_names = class_info("DomainNet")
        pred, y = dn["s_prob"].argmax(1), dn["label"]
        hard = np.argsort([np.mean(pred[y == c] == c) for c in range(126)])[:8]
        tsne(plt, "dn", ["source", "full"], dn_names, "DomainNet-126 S→P", 1000, list(hard), lines)
    dynamics(plt, "oh", ["full", "minusA", "minusB", "vsfot"], "Office-Home Ar→Cl", lines)
    dynamics(plt, "vd", ["full", "minusA", "minusB", "minusC", "vsfot"], "VisDA-C S→R", lines)
    visda_class(plt, vd_names, lines)
    visda_cm(plt, vd_names)
    open(os.path.join(OUT, "numbers.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n图在 {OUT}/：" + " ".join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "fig_*.pdf")))))


if __name__ == "__main__":
    {"extract": extract, "plot": plot}[sys.argv[1]]()
