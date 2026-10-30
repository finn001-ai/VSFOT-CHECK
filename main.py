import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import sys
import clip,copy
import torch
import argparse
from utils.Loss import *
from utils.Project_Record import Project
from utils.Dataset import getDataloaderSFDA5X
from utils.Other import backup_files_and_folders, seed_everything, load_class_names, check_class_names
from utils.Evaluate import modeleval, modeleval_str, lineClassAcc
from model.network import resnet, classifier
from utils.otUse import getGamma, getGamma_self
from utils.MemoryBank import MemoryBank
from utils.adapterVIT import clipAdapter
from utils.promptLearner import PromptLearner
from model.simsiam import SimSiam, fastSimsiam
import datetime
import json
from tqdm import tqdm
import torch.nn.functional as F
import random
import math
import numpy as np


IMAGEFILE = "dataset"    # 相对项目根目录；图片通过软链接放在 dataset/<数据集>/<域>/ 下

def get_rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def set_rng_state(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


def eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, loader, device, text_fn=None):
    """训练动态：学生 / 老师（带 adapter 的 CLIP）在目标域全部图片上的预测、熵和标签（标签只用于记录）"""
    rng_state = get_rng_state()          # 遍历 DataLoader 会消耗随机数，恢复后训练过程和不记录时完全一致
    feature_net.eval(); CLIP_model.eval()
    N = len(loader.dataset)
    out = {k: torch.zeros(N, dtype=torch.long, device=device) for k in ("s_pred", "t_pred", "label")}
    out.update({k: torch.zeros(N, device=device) for k in ("s_ent", "t_ent")})
    with torch.no_grad():
        tf = text_fn() if text_fn is not None else CLIP_model.encode_text(cls_text)      # --t_prompt：用学到的提示
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for item in loader:
            idx = item["index"].to(device)
            ps = torch.softmax(classifier_net(feature_net(item["img"].to(device).float())), dim=1)
            vf = CLIP_model.encode_image(item["clip_img"].to(device).float())
            pt = torch.softmax(100 * (vf / vf.norm(dim=-1, keepdim=True)) @ tf.T, dim=1)
            out["s_pred"][idx], out["t_pred"][idx] = ps.argmax(1), pt.argmax(1)
            out["s_ent"][idx] = -(ps * torch.log(ps + 1e-8)).sum(1)
            out["t_ent"][idx] = -(pt * torch.log(pt + 1e-8)).sum(1)
            out["label"][idx] = item["label"].to(device)
    CLIP_model.train()                   # 训练循环每步开头会把 feature_net 设回 train
    set_rng_state(rng_state)
    return out


def log_dynamics(path, iter_num, epoch, cur, prev, acc, lr):
    """把一行训练动态追加到 dynamics.csv；acc 是本 epoch 训练中的损失 / 伪标签统计（可为空）"""
    y = cur["label"]
    s_ok, t_ok = cur["s_pred"] == y, cur["t_pred"] == y
    row = {
        "iter": iter_num, "epoch": epoch, "lr": lr,
        "student_acc": s_ok.float().mean().item(), "teacher_acc": t_ok.float().mean().item(),
        "agree": (cur["s_pred"] == cur["t_pred"]).float().mean().item(),
        "either_acc": (s_ok | t_ok).float().mean().item(),
        "both_wrong": int((~s_ok & ~t_ok).sum()),
        "student_flip": (cur["s_pred"] != prev["s_pred"]).float().mean().item() if prev else float("nan"),
        "teacher_flip": (cur["t_pred"] != prev["t_pred"]).float().mean().item() if prev else float("nan"),
        "student_ent": cur["s_ent"].mean().item(), "teacher_ent": cur["t_ent"].mean().item(),
    }
    n = acc.get("n", 0)
    for k in ("loss_align", "loss_ctr", "loss_iml", "loss_vil"):
        row[k] = acc[k] / n if n else float("nan")
    m = acc.get("m", 0)
    row["pl_gamma_acc"] = acc["pl_gamma"] / m if m else float("nan")     # 对齐用的计划 gamma1 每列 argmax 的准确率
    row["pl_align_acc"] = acc["pl_align"] / m if m else float("nan")     # 对齐用的 CLIP 概率（近邻平滑后）argmax 的准确率
    row["mgva_keep"] = acc.get("keep", 0) / m if m else float("nan")      # MGVA 教老师时保留的样本比例
    row["mgva_keep_acc"] = acc["keep_ok"] / acc["keep"] if acc.get("keep") else float("nan")   # 保留样本上学生 top-1 的准确率
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fp:
        if new:
            fp.write(",".join(row.keys()) + "\n")
        fp.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
    return row


def save_samples(path, hist, iter_num, cur):
    """--log_samples：把这次评估的逐样本预测追加到 hist，并整体写入 samples.npz（每行一次评估，列为目标域样本）"""
    hist.setdefault("iter", []).append(iter_num)
    for k in ("s_pred", "t_pred", "s_ent", "t_ent"):
        hist.setdefault(k, []).append(cur[k].cpu().numpy())
    np.savez_compressed(path, label=cur["label"].cpu().numpy(), **{k: np.array(v) for k, v in hist.items()})


def eval_rich(feature_net, classifier_net, CLIP_model, cls_text, loader, device, text_fn=None, with_feat=False):
    """--log_rich：学生 / 老师在目标域全部图片上的完整概率（原图和水平翻转各一份），可选两边的 L2 归一化特征；不影响训练过程"""
    rng_state = get_rng_state()
    feature_net.eval(); CLIP_model.eval()
    N = len(loader.dataset)
    out = {}
    with torch.no_grad():
        tf = text_fn() if text_fn is not None else CLIP_model.encode_text(cls_text)
        tf = F.normalize(tf.float(), dim=-1)
        for item in loader:
            idx = item["index"].to(device)
            for flip, sfx in ((False, ""), (True, "_flip")):
                x, xc = item["img"].to(device).float(), item["clip_img"].to(device).float()
                if flip:
                    x, xc = torch.flip(x, dims=[3]), torch.flip(xc, dims=[3])
                f = feature_net(x)
                ps = torch.softmax(classifier_net(f), dim=1)
                vf = F.normalize(CLIP_model.encode_image(xc).float(), dim=-1)
                pt = torch.softmax(100 * vf @ tf.T, dim=1)
                vals = {"s_prob" + sfx: ps, "t_prob" + sfx: pt}
                if with_feat and not flip:
                    vals.update({"s_feat": F.normalize(f.float(), dim=-1), "t_feat": vf})
                for k, v in vals.items():
                    if k not in out:
                        out[k] = torch.zeros(N, v.shape[1], dtype=torch.float16, device=device)
                    out[k][idx] = v.half()
    CLIP_model.train()
    set_rng_state(rng_state)
    return {k: v.cpu().numpy() for k, v in out.items()}


def save_rich(root, hist, iter_num, cur, label):
    """概率每次都存（rich.npz，[评估次数, N, C]）；特征只在 with_feat 的评估存，单独一个文件 rich_feat_<iter>.npz"""
    hist.setdefault("iter", []).append(iter_num)
    for k in ("s_prob", "t_prob", "s_prob_flip", "t_prob_flip"):
        hist.setdefault(k, []).append(cur[k])
    np.savez(os.path.join(root, "rich.npz"), label=label, **{k: np.array(v) for k, v in hist.items()})
    if "s_feat" in cur:
        np.savez(os.path.join(root, f"rich_feat_{iter_num:05d}.npz"), s_feat=cur["s_feat"], t_feat=cur["t_feat"])


def init_feat_bank(feature_net, loader, device):
    """学生特征库：目标域每张图的学生特征（L2 归一化），用测试变换做一遍前向初始化"""
    # 遍历 DataLoader 会消耗随机数，先保存再恢复，保证之后的训练随机序列和不开 --knn_k 时完全一致
    rng_state = get_rng_state()
    feature_net.eval()
    bank = None
    with torch.no_grad():
        for item in tqdm(loader, desc="init feat bank"):
            feat = F.normalize(feature_net(item["img"].to(device).float()), dim=1)
            if bank is None:
                bank = torch.zeros(len(loader.dataset), feat.shape[1], device=device)
            bank[item["index"].to(device)] = feat
    set_rng_state(rng_state)
    return bank


def knn_smooth(feat, index, feat_bank, prob_bank, own_prob, k):
    """(自身概率 + 学生特征空间中 k 个近邻在 prob_bank 中的概率平均) / 2，近邻不含自身"""
    sim = F.normalize(feat, dim=1) @ feat_bank.T                          # [B, N]
    sim[torch.arange(len(index), device=sim.device), index] = -2.0        # 排除自身
    neighbors = sim.topk(k, dim=1).indices                                # [B, k]
    return (own_prob + prob_bank[neighbors].mean(dim=1)) / 2


def init_src_bank(feature_net, classifier_net, loader, device):
    """--src_expert：冻结源模型（训练开始前的学生）在目标域每张图上的概率，用测试变换做一遍前向；之后不再更新"""
    rng_state = get_rng_state()          # 同 init_feat_bank：恢复随机数，训练随机序列不受影响
    feature_net.eval()
    bank = None
    with torch.no_grad():
        for item in tqdm(loader, desc="init src bank"):
            prob = torch.softmax(classifier_net(feature_net(item["img"].to(device).float())), dim=1)
            if bank is None:
                bank = torch.zeros(len(loader.dataset), prob.shape[1], device=device)
            bank[item["index"].to(device)] = prob
    set_rng_state(rng_state)
    return bank


def entropy_poe(p, q, ref_p=None, ref_q=None):
    """按熵加权的专家乘积（COSMO / ProDe 的源模型去噪思路）：r ∝ p^a · q^(1-a)，a = e^{-H(p)} / (e^{-H(p)} + e^{-H(q)})，熵越小权重越大
    ref_p / ref_q（--poe_weight rank）：两个专家各自全部样本熵的升序排列。熵换成「在自己模型内的百分位 × log C」再比较，
    消掉两个模型整体校准的差别（CLIP 的 logit 乘 100，老师的熵整体比学生小，绝对熵下共识几乎总听老师，Office-Home 上学生更准时也一样）"""
    hp = -(p * torch.log(p + 1e-8)).sum(1, keepdim=True)
    hq = -(q * torch.log(q + 1e-8)).sum(1, keepdim=True)
    if ref_p is not None:
        logc = math.log(p.size(1))
        hp = torch.searchsorted(ref_p, hp.contiguous()).float() / len(ref_p) * logc
        hq = torch.searchsorted(ref_q, hq.contiguous()).float() / len(ref_q) * logc
    a = torch.exp(-hp) / (torch.exp(-hp) + torch.exp(-hq))
    return torch.softmax(a * torch.log(p + 1e-8) + (1 - a) * torch.log(q + 1e-8), dim=1)


def kmeans_cos(x, k, restarts=5, iters=50, seed=0):
    """余弦 k-means（x 已 L2 归一化），k-means++ 初始化，取簇内相似度和最大的一次；返回 [n] 簇号（同 tools/diag_split.py）"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_score = None, -1e9
    n = x.shape[0]
    for _ in range(restarts):
        cent = x[torch.randint(n, (1,), generator=g)].clone()
        for _ in range(1, k):
            d = (1 - (x @ cent.T).max(1).values).clamp(min=0) ** 2
            prob = (d / d.sum()).cpu() if d.sum() > 0 else torch.full((n,), 1.0 / n)
            cent = torch.cat([cent, x[torch.multinomial(prob, 1, generator=g)]])
        for _ in range(iters):
            a = (x @ cent.T).argmax(1)
            new = torch.stack([F.normalize(x[a == j].mean(0), dim=0) if (a == j).any() else cent[j] for j in range(k)])
            if torch.allclose(new, cent):
                break
            cent = new
        score = (x @ cent.T).max(1).values.sum().item()
        if score > best_score:
            best, best_score = a.clone(), score
    return best


def bn_recal_eval(args, feature_net, classifier_net, loader):
    """--eval_bn_recal（诊断，不改训练）：复制学生，BN 运行统计量清零后在目标域全部图片（测试变换）上累计平均重估，再评估。
    VisDA 全量两次在 iter 24220（学习率 1e-4）train 类都掉到 58%、下一次恢复：怀疑是最近几个 batch 把 BN 统计量带偏，而不是权重"""
    rng_state = get_rng_state()
    net = copy.deepcopy(feature_net)
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None          # momentum=None：累计平均
    net.train()
    with torch.no_grad():
        for item in loader:
            net(item["img"].to(args.device).float())
    out = modeleval(args, net, classifier_net, loader)
    del net
    set_rng_state(rng_state)
    return out


def split_relabel(feature_net, classifier_net, CLIP_model, text_fn, cls_text, loader, device, C, k, theta, min_size=5):
    """--split_at：在老师 CLIP 特征空间里把每个学生预测类拆成 k 个子簇；子簇平均概率（(学生 + 老师) / 2）里预测类以外的第一名 r
    满足 m[r] / m[c] ≥ theta、且子簇中心离 r 类中心（各类高置信一半样本的均值）比离 c 类中心更近时，整个子簇改名为 r。
    返回 [N] 改名后的类（-1 = 不改）和统计（标签只用于记录）。离线诊断：Ar→Pr 进度 5% 快照 +2.79（纠正 160 / 改坏 36），见 tools/diag_split.py"""
    rng_state = get_rng_state()
    feature_net.eval(); CLIP_model.eval()
    N = len(loader.dataset)
    ps, pt, vf, y = torch.zeros(N, C, device=device), torch.zeros(N, C, device=device), None, torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        tf = text_fn() if text_fn is not None else CLIP_model.encode_text(cls_text)
        tf = F.normalize(tf.float(), dim=-1)
        for item in loader:
            idx = item["index"].to(device)
            ps[idx] = torch.softmax(classifier_net(feature_net(item["img"].to(device).float())), dim=1)
            f = F.normalize(CLIP_model.encode_image(item["clip_img"].to(device).float()).float(), dim=-1)
            if vf is None:
                vf = torch.zeros(N, f.shape[1], device=device)
            vf[idx] = f
            pt[idx] = torch.softmax(100 * f @ tf.T, dim=1)
            y[idx] = item["label"].to(device)
        avg = (ps + pt) / 2
        pred, conf = ps.argmax(1), avg.max(1).values
        cen = torch.zeros(C, vf.shape[1], device=device)
        for c in range(C):
            idx = (pred == c).nonzero(as_tuple=True)[0]
            if len(idx):
                cen[c] = F.normalize(vf[idx[conf[idx] >= conf[idx].median()]].mean(0), dim=0)
        fix = torch.full((N,), -1, dtype=torch.long, device=device)
        for c in range(C):
            idx = (pred == c).nonzero(as_tuple=True)[0]
            if len(idx) < 2 * min_size:
                continue
            a = kmeans_cos(vf[idx], k).to(device)
            for j in range(k):
                gi = idx[a == j]
                if len(gi) < min_size:
                    continue
                m = avg[gi].mean(0)
                m_c = m.clone(); m_c[c] = -1
                r = m_c.argmax().item()
                gc = F.normalize(vf[gi].mean(0), dim=0)
                if m[r] / m[c] >= theta and gc @ cen[r] > gc @ cen[c]:
                    fix[gi] = r
    feature_net.train(); CLIP_model.train()
    set_rng_state(rng_state)
    ch = fix >= 0
    stats = {"changed": int(ch.sum()), "fixed": int((ch & (fix == y) & (pred != y)).sum()),
             "broke": int((ch & (fix != y) & (pred == y)).sum()), "student": (pred == y).float().mean().item(),
             "after": torch.where(ch, fix, pred).eq(y).float().mean().item()}
    return fix, stats


def start(args, target_dataloader, CLIP_model_raw):
    network = torch.load(os.path.join("runs", args.sourceStorage, f"Source_{args.source.lower()}", "Source.pt"), weights_only=False)
    feature_net = resnet(res_name=args.arch).to(args.device)
    feature_net.load_state_dict(network["best_feature"])
    classifier_net = classifier(class_num = args.class_num).to(args.device)
    classifier_net.load_state_dict(network["best_classifier"])
    iter_num = 0
    now_epoch = 0
    epoch_iter_num = len(target_dataloader["train"])
    eval_gap = epoch_iter_num
    max_iter = args.epochs * epoch_iter_num

    param_group = [{'params': feature_net.backbone.parameters(), 'lr': args.lr * 0.1},
                   {'params': feature_net.feat_bottleneck.parameters(), 'lr': args.lr},
                   {'params': classifier_net.parameters(), 'lr':args.lr * args.cls_lr_mult}]   # --cls_lr_mult 默认 0.1 = 原方法
    model_optimizer = torch.optim.SGD(param_group)
    model_LR = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, max_iter, eta_min=0, last_epoch=- 1)

    CLIP_model = clipAdapter(CLIP_model_raw, args.device, adapterCount = args.adaptercount)
    clip_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CLIP_model.parameters()), lr=5e-5)

    class_prompt = [ f'a photo of a {item}' for item in args.class_info]
    cls_text = clip.tokenize(class_prompt).to(args.device)
    # --t_prompt：老师的文本端加可学习的提示上下文（4 个所有类共享的向量，用 "a photo of a" 的词嵌入初始化，初始输出与固定提示完全相同），
    #   和 adapter 一起由 MGVA 更新（DIFO / ProDe / COSMO 等同类方法的标准组件）；H1：学生的缺口继承自老师固定的文本原型分类头
    #   --no_adapter：只学提示上下文，图像侧 adapter 不更新（同 COSMO）
    #   --t_prompt_opt sgd：提示上下文用 DIFO / ProDe / DIFO++ 公开代码的优化器（SGD，动量 0.9，weight decay 5e-4，学习率 = 主学习率 × 0.1），单独一个优化器
    prompt_learner, text_fn, prompt_optimizer = None, None, None
    if args.t_prompt:
        prompt_learner = PromptLearner(CLIP_model, cls_text, n_ctx=args.t_prompt_nctx).to(args.device)
        text_fn = prompt_learner
        if args.t_prompt_opt == "sgd":
            prompt_optimizer = torch.optim.SGD([prompt_learner.ctx], lr=args.lr * 0.1, momentum=0.9, weight_decay=5e-4)
            if args.no_adapter:                     # 只学提示上下文：adapter 冻结，不再需要 Adam
                for prm in CLIP_model.parameters():
                    prm.requires_grad = False
                clip_optimizer = None
        elif args.no_adapter:
            for prm in CLIP_model.parameters():
                prm.requires_grad = False
            clip_optimizer = torch.optim.Adam([prompt_learner.ctx], lr=args.t_prompt_lr)
        else:
            clip_optimizer.add_param_group({"params": [prompt_learner.ctx], "lr": args.t_prompt_lr})
    encode_cls_text = text_fn if text_fn is not None else (lambda: CLIP_model.encode_text(cls_text))

    MB = MemoryBank(target_dataloader["test"].dataset.__len__(), args.class_num, device=args.device)
    with torch.no_grad():
        cls_text_features = CLIP_model_raw.encode_text(cls_text)
        MB.initialize(target_dataloader["test"], CLIP_model_raw, feature_net, classifier_net, cls_text_features, args.device)
    # --clsw_mode prior：类均衡的目标分布 = CLIP 零样本在目标域上的平均概率（训练开始时算一次，之后固定）
    clsw_prior = MB.vil_bank.mean(dim=0).clone()
    # --clsw_mode oracle：诊断上界，目标分布换成目标域真实类频率（用了标签，只用来判断「类先验估计」这条路的空间，不是方法）
    if args.clsw_mode == "oracle":
        y_all = torch.tensor([int(l) for _, l in target_dataloader["test"].dataset.imgs], device=args.device)
        clsw_prior = torch.bincount(y_all, minlength=args.class_num).float() / len(y_all)
    # --mgva_gate agree：冻结的零样本 CLIP 预测（训练开始时的 vil_bank，之后 vil_bank 会被当前老师覆盖）
    zs_pred = MB.vil_bank.argmax(dim=1).clone() if args.mgva_gate == "agree" else None
    # --t_proto：老师的类原型（原为固定的文本嵌入）变成可学习参数，用文本嵌入初始化，和 adapter 一起由 MGVA 更新
    #   H1（archive/2026_10_20_h1）：学生「探针 − 实际」与老师几乎一一对应，学生继承的是固定文本原型分类头的错误
    t_proto, t_proto_init = None, None
    if args.t_proto:
        with torch.no_grad():
            t_proto_init = F.normalize(CLIP_model.encode_text(cls_text).float(), dim=1)
        t_proto = torch.nn.Parameter(t_proto_init.clone())
        clip_optimizer.add_param_group({"params": [t_proto]})      # 学习率同 adapter（5e-5）
    # --knn_k > 0：在学生特征空间里用近邻平滑伪标签（静态诊断见 archive/2026_10_16_diag_struct）
    feat_bank = init_feat_bank(feature_net, target_dataloader["test"], args.device) if args.knn_k > 0 else None
    # --src_expert：冻结源模型的概率库（必须在训练开始前、学生还等于源模型时计算）
    src_bank = init_src_bank(feature_net, classifier_net, target_dataloader["test"], args.device) if args.src_expert else None
    SSNB = SimSiam(feature_net).to(args.device)
    # --split_at：混淆类拆分纠正（改名结果在之后的训练里覆盖该样本的对齐计划和老师目标）
    split_fix = None

    model_cm, model_oa, model_aa, model_class_acc = modeleval(args, feature_net, classifier_net, target_dataloader["test"])
    ModellineAcc = lineClassAcc(args, model_class_acc, 0, header=True)
    Project.log(ModellineAcc, filename=f"ModellineAcc.txt", is_print=False, haveTime=False)
    Project.log(f"\n\n\n ==>iter: {iter_num}  " + modeleval_str(model_oa,model_aa,model_cm, args.class_info), filename=f"Detail.txt", is_print = False)
    best_acc = model_oa

    IML = InformationMaximizationLoss(trade_off=args.imloss, ent_weight=args.iml_ent)
    # 老师侧（2026-10-20，scripts/run_teacher.sh）：组件审计中老师不更新 −4.15，而老师唯一的学习信号是学生当前这一批的 top-3
    #   --t_ema：老师学的学生概率换成逐样本跨 epoch 的动量平均（初始 = 源模型概率），少吸收学生的临时错误
    #   --t_strong：老师在强增强视图上也学同一个目标（损失取平均），给老师一个视图一致性约束
    #   --t_im：老师的损失加上它自己预测的 IM（熵 + diversity），一个不依赖学生的学习信号
    t_ema_bank = MB.model_bank.clone() if args.t_ema else None
    # --s_ema：学生（特征网络 + 分类头）权重和 BN 统计量的指数滑动平均，时间常数 --s_ema 个 epoch；只用于评估，不参与训练（训练与不加时逐位相同）
    #   VisDA 全量两次在 iter 24220 train 类瞬时掉到 58%（BN 统计量被最近的 batch 带偏，archive/2026_10_23_cross），EMA 把这种抖动平滑掉
    ema_nets, ema_decay = None, 0.0
    if args.s_ema > 0:
        ema_nets = (copy.deepcopy(feature_net), copy.deepcopy(classifier_net))
        ema_decay = 1.0 - 1.0 / max(args.s_ema * epoch_iter_num, 1.0)
    IML_t = InformationMaximizationLoss(trade_off=1.0) if args.t_im > 0 else None
    # 强增强视图是 ImageNet 归一化的 224 图，换成 CLIP 的归一化（同尺寸，不需要重新读图，不消耗额外随机数）
    in_mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    in_std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=args.device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=args.device).view(1, 3, 1, 1)

    # 诊断用：在指定训练进度保存快照（默认不保存），并可提前停止（默认跑完）
    # 学习率调度仍按完整的 max_iter 计算，所以快照和完整训练时同一进度的状态一致
    # 进度 1.0 的快照存在最后一步（max_iter - 1）开始前：循环到不了 max_iter
    snapshot_iters = {min(int(p * max_iter), max_iter - 1): p for p in args.snapshot_at}
    stop_iter = int(args.stop_at * max_iter)
    # 共识塌缩实验：默认 1.0 = 永不触发（原方法）
    freeze_teacher_iter = int(args.freeze_teacher_at * max_iter)    # 从这一步起老师（CLIP adapter）不再更新
    vil_low_iter = int(args.vil_after * max_iter)                    # 从这一步起 MGVA 的学习率乘以 --vil_low
    knn_self_iter = int(args.knn_self_after * max_iter)              # 从这一步起近邻平滑混入学生概率
    align_agree_iter = int(args.align_agree_after * max_iter)
    src_expert_iter = int(args.src_expert_until * max_iter)         # --src_expert：到这一步为止对齐目标混入冻结源模型        # 从这一步起只在师生一致的样本上做对齐

    # --log_dynamics：每个 epoch 记录学生 / 老师的准确率、一致率、翻转率、熵、伪标签准确率和各项损失（dynamics.csv）
    dyn_path = os.path.join(Project.root_path, "dynamics.csv")
    dyn_prev, dyn_acc = None, {}
    # --log_samples：每次记录训练动态时，同时保存逐样本的学生 / 老师预测和熵（samples.npz，用于状态转移分析）
    samples_path, samples_hist = os.path.join(Project.root_path, "samples.npz"), {}
    if args.log_dynamics:
        dyn_prev = eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, target_dataloader["test"], args.device, text_fn)
        if args.log_samples:
            save_samples(samples_path, samples_hist, 0, dyn_prev)
        r = log_dynamics(dyn_path, 0, 0, dyn_prev, None, {}, model_optimizer.param_groups[1]["lr"])
        if args.log_rich:
            rich_hist, rich_n = {}, 0
            rich_label = dyn_prev["label"].cpu().numpy()
            save_rich(Project.root_path, rich_hist, 0, eval_rich(feature_net, classifier_net, CLIP_model, cls_text, target_dataloader["test"],
                                                                 args.device, text_fn, with_feat=True), rich_label)
        tqdm.write(f"[dyn] iter 0  学生 {r['student_acc']:.4f}  老师 {r['teacher_acc']:.4f}  一致率 {r['agree']:.3f}")

    # 进度条代替逐步打印；每一步的损失仍完整写入 run_log.txt
    bar = tqdm(total=min(stop_iter, max_iter), desc=f"{args.source}->{args.target}", dynamic_ncols=True, mininterval=1.0)

    while iter_num < max_iter:
        if iter_num in snapshot_iters:
            p = snapshot_iters[iter_num]
            torch.save({
                "progress": p, "iter": iter_num, "max_iter": max_iter,
                "args": vars(args),
                "imgs": target_dataloader["test"].dataset.imgs,     # 记忆库索引对应的图片顺序
                "feature": feature_net.state_dict(),
                "classifier": classifier_net.state_dict(),
                "adapter": {k: v for k, v in CLIP_model.state_dict().items() if "adapter" in k},
                "prompt_ctx": prompt_learner.ctx.detach().cpu() if prompt_learner is not None else None,
                "vil_bank": MB.vil_bank.cpu(),
                "model_bank": MB.model_bank.cpu(),
            }, os.path.join(Project.root_path, f"snapshot_p{round(p * 100):03d}.pt"))
            Project.log(f"Snapshot saved at progress {p} (iter {iter_num})", is_print=False)
            tqdm.write(f"Snapshot saved at progress {p} (iter {iter_num})")
        # --split_at：到该进度时做一次混淆类拆分纠正；--split_until > --split_at 时之后每个 epoch 重做一次，直到该进度
        if args.split_at < 1.0 and iter_num >= int(args.split_at * max_iter) and iter_num <= int(max(args.split_until, args.split_at) * max_iter) \
                and (split_fix is None or (iter_num - int(args.split_at * max_iter)) % epoch_iter_num == 0):
            split_fix, st = split_relabel(feature_net, classifier_net, CLIP_model, text_fn, cls_text, target_dataloader["test"],
                                          args.device, args.class_num, args.split_k, args.split_theta)
            msg = (f"[split] iter {iter_num}  改名 {st['changed']}（纠正 {st['fixed']} / 改坏 {st['broke']}），"
                   f"学生 {st['student']:.4f} → {st['after']:.4f}")
            Project.log(msg, is_print=False)
            tqdm.write(msg)
        # --split_release：到该进度撤掉改名（之后回到原来的对齐计划和老师目标）
        if split_fix is not None and iter_num >= int(args.split_release * max_iter):
            split_fix = None
            Project.log(f"[split] iter {iter_num}  撤掉改名", is_print=False)
            tqdm.write(f"[split] iter {iter_num}  撤掉改名")
        if iter_num >= stop_iter:
            Project.log(f"Stop at progress {args.stop_at} (iter {iter_num})", is_print=False)
            tqdm.write(f"Stop at progress {args.stop_at} (iter {iter_num})")
            break
        feature_net.train()
        try:
            item = next(iter_target)
        except:
            iter_target = iter(target_dataloader["train"])
            item = next(iter_target)
            now_epoch += 1
        finally:
            weak_img = item["img"].to(args.device).float()
            clip_img = item["clip_img"].to(args.device).float()
            index_img = item["index"].to(args.device)
            strong_img = item["strong_img"].to(args.device).float()
            strong2_img = item["strong2_img"].to(args.device).float()
        vilMemoryMean, modelMemoryMean = MB.get(index_img)

        ################################################
        with torch.no_grad():
            modelFeat = feature_net(weak_img)                   # [B, FM]
            modelLogits = classifier_net(modelFeat)             # [B, C]
            modelProb = torch.softmax(modelLogits, dim=1)       # [B, C]

            vilImgFeat = CLIP_model.encode_image(clip_img)      # [B, FV]
            vilTextFeat = encode_cls_text()                     # [C, FV]（--t_prompt 时为学到的提示）
            vilImgFeat = vilImgFeat / vilImgFeat.norm(dim=-1, keepdim=True)         # [B, FV]
            vilTextFeat = vilTextFeat / vilTextFeat.norm(dim=-1, keepdim=True)      # [B, FC]
            if args.t_proto:                                                        # 老师给学生的预测也用学到的原型
                vilTextFeat = F.normalize(t_proto.detach(), dim=1)
            clipSim = vilImgFeat @ vilTextFeat.T                                    # [B, C]
            clipProb = torch.softmax(100*clipSim, dim = 1)                          # [B, C]
            # --t_flip：老师给学生的预测取原图与水平翻转图的概率平均（伪标签的多视图平均，老师本身不变）
            if args.t_flip:
                vilImgFeatF = F.normalize(CLIP_model.encode_image(torch.flip(clip_img, dims=[3])), dim=-1)
                clipProb = (clipProb + torch.softmax(100 * vilImgFeatF @ vilTextFeat.T, dim=1)) / 2
            # 对齐损失用的 CLIP 概率：开启 --knn_k 时换成学生特征空间近邻平滑后的版本（记忆库仍存原始 clipProb）
            # --knn_self_after：到该进度后，平滑用的自身概率和近邻概率都换成 (1 - w) · CLIP + w · 学生（w = --knn_self_w），
            #   让对齐目标在后期仍高于学生自己（静态诊断 p050：学生自身的近邻平滑比学生高 +1.0）；μ_p 仍取 CLIP 记忆库
            # --src_expert：对齐目标里的 CLIP 概率（自身和近邻的）换成 CLIP 与冻结源模型按熵加权的专家乘积；
            #   诊断：p000 时 Ar→Cl 有 407 个样本只有源模型判对，原方法训练开始后不再使用源模型
            # --poe_weight rank：老师 / 学生记忆库里全部样本的熵（升序），共识权重按各自模型内的百分位计算
            poe_ref = (None, None)
            if args.poe_weight == "rank" and args.consensus != "none":
                bank_ent = lambda b: torch.sort(-(b * torch.log(b + 1e-8)).sum(1)).values
                poe_ref = (bank_ent(MB.vil_bank), bank_ent(MB.model_bank))
            if args.src_expert and iter_num < src_expert_iter:
                clipAlign = entropy_poe(clipProb, src_bank[index_img])
                vilBankAlign = entropy_poe(MB.vil_bank, src_bank)
            # --consensus both：学生对齐的目标（自身和近邻的）换成老师与学生按熵加权的专家乘积（COSMO 式共享共识，更新前的预测，detach）
            elif args.consensus in ("both", "student"):
                clipAlign = entropy_poe(clipProb, modelProb, *poe_ref)
                vilBankAlign = entropy_poe(MB.vil_bank, MB.model_bank, *poe_ref)
            else:
                clipAlign, vilBankAlign = clipProb, MB.vil_bank
            if args.knn_k > 0 and iter_num >= knn_self_iter:
                w = args.knn_self_w
                alignProb = knn_smooth(modelFeat, index_img, feat_bank, (1 - w) * MB.vil_bank + w * MB.model_bank,
                                       (1 - w) * clipProb + w * modelProb, args.knn_k)
            elif args.knn_k > 0:
                alignProb = knn_smooth(modelFeat, index_img, feat_bank, vilBankAlign, clipAlign, args.knn_k)
            else:
                alignProb = clipAlign

            if args.align_mode == "ot":
                gamma1 = getGamma(args, alignProb.T, vilMemoryMean, args.otReg)     # [C, B]
            # 消融：不做 OT，直接用 CLIP 的标签当计划；每列质量 1/B，和 OT 计划的列边际一致，损失量级可比
            elif args.align_mode == "clip_soft":
                gamma1 = alignProb.T / alignProb.size(0)                              # [C, B]
            elif args.align_mode == "clip_hard":
                gamma1 = torch.nn.functional.one_hot(alignProb.argmax(dim=1), args.class_num).float().T / alignProb.size(0)
            # 消融：和 OT 用同一个代价 / 核（同 getGamma），只做列归一化，不做 Sinkhorn 的类别边际投影
            # ot - clip_sharp = 边际投影的作用；clip_sharp - clip_soft = 锐化的作用
            elif args.align_mode == "clip_sharp":
                feat_dist = 1 - alignProb.T                                          # [C, B]
                cost = feat_dist / feat_dist.max() - torch.log(alignProb.T + 1e-6)   # 同 getGamma 的 cost_matrix
                gamma1 = torch.softmax(-cost / args.otReg, dim=0) / alignProb.size(0)   # 每列 ∝ exp(-cost/ε)，质量 1/B

        ################################################
        modelFeatGradW = feature_net(weak_img)
        modelLogitsGradW = classifier_net(modelFeatGradW)
        modelProbGrad = torch.softmax(modelLogitsGradW, dim=1)
        _, cost_matrix = getGamma_self(args, modelProbGrad, modelFeatGradW, modelMemoryMean, classifier_net, reg=args.otReg, need_gamma=False)

        # --split_at：改过名的样本，对齐计划这一列换成改名后的类（质量 1/B，与计划其他列一致）
        if split_fix is not None:
            fb = split_fix[index_img]
            sm = fb >= 0
            if sm.any():
                gamma1[:, sm] = F.one_hot(fb[sm], args.class_num).float().T / gamma1.size(1)

        with torch.no_grad():
            total_cls = MB.model_bank.mean(dim=0)
            cls_freq = torch.clamp(total_cls, min=1e-4)
            class_weights = 1.0 / cls_freq
            # --no_clsw：消融，去掉按学生预测频率的倒数重加权（它假设类别均衡）
            if args.no_clsw:
                class_weights = torch.ones_like(class_weights)
        # --align_agree_after：到该进度后，学生预测和计划 argmax（重加权前）不一致的样本不再做对齐（列置 0，不重新归一化）
        #   分歧裁决分析：第 612 步之后分歧样本上学生判对的次数约为老师的 2 倍，无条件对齐等于让对的一方服从错的一方
        if iter_num >= align_agree_iter:
            gamma1 = gamma1 * (modelProb.argmax(dim=1) == gamma1.argmax(dim=0)).float().unsqueeze(0)
        # --clsw_mode：类频率重加权的形状（inv = 原方法 1/频率）。权重均值约为类别数，直接去掉会让对齐损失缩小约 65 倍（审计 noclsw 崩溃的原因），
        #   所以 const / sqrt 每一步都把「按计划质量加权的平均权重」调成和 inv 相同：损失量级不变，只改变类别之间的相对权重
        #   const = 所有类同一权重（不做均衡）；sqrt = 1/√频率（均衡推得轻一半）
        #   prior = CLIP 零样本平均概率 / 频率：把学生的预测分布推向 CLIP 估计的类分布，而不是均匀分布（Art 最不均衡，sqrt 在 Art 上 +1.2）
        # --clsw_mode self：进度 --clsw_self_at 之前同 inv（推向均匀，防止 CLIP 的类偏置塌缩）；到该进度时把学生记忆库的平均概率冻结为目标分布，
        #   之后推向这个分布。诊断（archive/2026_10_23_cross）：进度 0.25 起学生的预测分布与真实先验的 L1 只有均匀分布的 1/3 ~ 2/3
        if args.clsw_mode == "self" and iter_num == int(args.clsw_self_at * max_iter):
            clsw_prior = MB.model_bank.mean(dim=0).clone()
            # --clsw_self_em：冻结前学生是在「推向均匀」的目标下训练的，平均概率会把先验压向均匀（VisDA：car 真实 18.8%，冻结时只有 14.3%，
            #   最多 / 最少 3.3 vs 真实 5.1）。按标签偏移的 EM（Saerens et al. 2002，MLLS）把训练先验当作均匀，迭代到收敛重估目标先验
            if args.clsw_self_em:
                with torch.no_grad():
                    for _ in range(100):
                        q = MB.model_bank * clsw_prior
                        new_prior = (q / q.sum(dim=1, keepdim=True)).mean(dim=0)
                        if (new_prior - clsw_prior).abs().sum() < 1e-6:
                            break
                        clsw_prior = new_prior
                    clsw_prior = new_prior
            Project.log(f"[clsw self] iter {iter_num}  冻结目标分布：最多 / 最少 = {(clsw_prior.max() / clsw_prior.min()).item():.1f}", is_print=False)
        if args.clsw_mode != "inv" and not (args.clsw_mode == "self" and iter_num < int(args.clsw_self_at * max_iter)):
            with torch.no_grad():
                mass = gamma1.sum(dim=1)                                                         # [C] 本批计划在各类上的质量
                w = {"const": torch.ones_like(class_weights), "sqrt": class_weights.sqrt(),
                     "prior": clsw_prior * class_weights, "oracle": clsw_prior * class_weights, "self": clsw_prior * class_weights,
                     "pow": class_weights ** args.clsw_pow}[args.clsw_mode]
                class_weights = w * (mass @ class_weights) / (mass @ w + 1e-12)
        gamma1 = gamma1 * class_weights.unsqueeze(1)
        loss_align = torch.sum(gamma1 * cost_matrix)
        # --align_view：学生在哪个视图上对齐老师的计划（老师的计划始终来自 CLIP 对 clip_img 的判断）
        #   weak（默认，原方法）：学生看弱增强图像；strong：学生看强增强图像（Noisy Student：
        #   输入和老师不同，最优解不再是复制老师）；both：两者取平均
        if args.align_view != "weak":
            modelFeatGradS = feature_net(strong_img)
            modelProbGradS = torch.softmax(classifier_net(modelFeatGradS), dim=1)
            _, cost_matrix_s = getGamma_self(args, modelProbGradS, modelFeatGradS, modelMemoryMean, classifier_net, reg=args.otReg, need_gamma=False)
            loss_align_s = torch.sum(gamma1 * cost_matrix_s)
            loss_align = loss_align_s if args.align_view == "strong" else (loss_align + loss_align_s) / 2

        loss_ctr = fastSimsiam(strong_img, strong2_img, SSNB)

        loss_iml = IML(modelProbGrad)

        progress = iter_num / max_iter
        weight_align = min(1.0, progress * 2)  # [1 -> 2]
        # --align_weight：对齐权重的进度曲线。ramp = 原方法（0 → 1，进度 0.5 加满）；
        #   const = 一直为 1（老师更准的早期就全力对齐）；cap = min(2·进度, --align_cap)（交叉点之后不再加大老师 → 学生的力度）
        #   依据：老师只在前约 7.5% 的训练里比学生准，而原方法的权重在那之后才加满；锁死样本 220 → 645 → 854 → 970（p000 / 025 / 050 / 100）
        if args.align_weight == "const":
            weight_align = 1.0
        elif args.align_weight == "cap":
            weight_align = min(args.align_cap, progress * 2)
        # --align_norm_c：对齐损失乘了 1/频率的类权重（均值约为类别数 C），量级随 C 线性变化（Align / C：OH 1.13、VisDA 0.93、DN 1.36，
        #   IML 基本不变），同一个权重曲线在 VisDA 上太弱、在 DN 上太强。乘以 参考 C / C，参考取定 cap 时的 Office-Home（65）：OH 完全不变
        if args.align_norm_c > 0:
            weight_align = weight_align * args.align_norm_c / args.class_num

        loss1 = args.weightalign * weight_align * loss_align + \
                args.weightctr * loss_ctr + \
                args.weightiml * loss_iml
        # --s_ce：学生在弱 / 强视图上对共识（老师 × 学生专家乘积）argmax 的硬交叉熵（COSMO：去掉硬 CE 86.4 → 79.7）
        if args.s_ce > 0:
            with torch.no_grad():
                ce_target = entropy_poe(clipProb, modelProb).argmax(dim=1)
            loss_sce = F.cross_entropy(modelLogitsGradW, ce_target)
            if args.align_view != "weak":
                loss_sce = (loss_sce + F.cross_entropy(classifier_net(modelFeatGradS), ce_target)) / 2
            loss1 = loss1 + args.s_ce * loss_sce
        # 方向 C：学生自训练（不经过老师）。线性探针 95.6 vs 学生 92：表示里有、分类头没用上，用目标域结构正则分类头
        # --s_fm：FixMatch —— 弱视图上学生（更新前）置信度 ≥ --fm_tau 的样本，强视图上对其 argmax 做交叉熵（除以批大小）
        if args.s_fm > 0:
            with torch.no_grad():
                fm_conf, fm_target = modelProb.max(dim=1)
                fm_mask = (fm_conf >= args.fm_tau).float()
            logitsS = classifier_net(modelFeatGradS) if args.align_view != "weak" else classifier_net(feature_net(strong_img))
            loss_fm = (F.cross_entropy(logitsS, fm_target, reduction="none") * fm_mask).mean()
            loss1 = loss1 + args.s_fm * loss_fm
        # --s_mix：批内 mixup —— 弱视图图像按 λ ~ Beta(α, α) 混合，目标为两张图学生（更新前）argmax 的同比例混合
        if args.s_mix > 0:
            with torch.no_grad():
                lam = float(np.random.beta(args.mix_alpha, args.mix_alpha))
                lam = max(lam, 1 - lam)
                perm = torch.randperm(weak_img.size(0), device=weak_img.device)
                y1 = F.one_hot(modelProb.argmax(dim=1), args.class_num).float()
                mix_target = lam * y1 + (1 - lam) * y1[perm]
            mix_logits = classifier_net(feature_net(lam * weak_img + (1 - lam) * weak_img[perm]))
            loss_mix = -(mix_target * F.log_softmax(mix_logits, dim=1)).sum(dim=1).mean()
            loss1 = loss1 + args.s_mix * loss_mix
        # --s_aad：AaD（Yang et al. 2022）—— 吸引：学生预测与它在学生特征库里 k 个近邻（记忆库中的学生概率）的点积；
        #   分散：与批内其他样本的点积，权重 (1 + 10·进度)^(−0.75) 衰减。直接让分类头服从学生特征的近邻结构
        if args.s_aad > 0:
            with torch.no_grad():
                sim = F.normalize(modelFeat, dim=1) @ feat_bank.T
                sim[torch.arange(len(index_img), device=sim.device), index_img] = -2.0
                nbr_prob = MB.model_bank[sim.topk(args.knn_k, dim=1).indices]                 # [B, k, C]
            loss_att = -(modelProbGrad.unsqueeze(1) * nbr_prob).sum(dim=2).mean()
            pp = modelProbGrad @ modelProbGrad.T
            loss_dis = (pp.sum() - pp.diag().sum()) / modelProbGrad.size(0)
            loss1 = loss1 + args.s_aad * (loss_att + (1 + 10 * progress) ** (-0.75) * loss_dis)

        model_optimizer.zero_grad()
        loss1.backward()
        model_optimizer.step()
        if ema_nets is not None:
            with torch.no_grad():
                for ema_m, m in zip(ema_nets, (feature_net, classifier_net)):
                    for pe, pm in zip(ema_m.parameters(), m.parameters()):
                        pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
                    for be, bm in zip(ema_m.buffers(), m.buffers()):
                        if be.dtype.is_floating_point:
                            be.mul_(ema_decay).add_(bm, alpha=1 - ema_decay)
                        else:
                            be.copy_(bm)

        #########################################################

        clipFeatWGrad = CLIP_model.encode_image(clip_img)
        clipFeatWGrad = clipFeatWGrad / clipFeatWGrad.norm(dim=-1, keepdim=True)

        clipTextFeatGrad = encode_cls_text()
        clipTextFeatGrad = clipTextFeatGrad / clipTextFeatGrad.norm(dim=-1, keepdim=True)
        if args.t_proto:
            clipTextFeatGrad = F.normalize(t_proto, dim=1)
        # --ct_prompt：老师拆成两路。adapter（图像侧）照原方法学学生 top-k（MGVA，文本特征 detach）；提示上下文（文本侧）学老师与学生的专家乘积共识
        #   （图像特征 detach）。依据：adapter 老师学共识 ≈ 拟合自己，CLIP 偏差锁死、整类被吞（archive/2026_10_28_q1028）；只学提示 + 共识时
        #   没有整类被吞（Pr2Cl Clipboards 80 vs 0），但去掉 adapter 后老师太弱（→Ar −3 ~ −4，run_pc.sh）。共识只能经低容量的共享上下文作用到类别层面
        ctxTextFeat = None
        if args.ct_prompt:
            ctxTextFeat, clipTextFeatGrad = clipTextFeatGrad, clipTextFeatGrad.detach()

        logits_clip = (100.0 * clipFeatWGrad @ clipTextFeatGrad.T)

        # --knn_mgva：老师（CLIP adapter）学的目标换成学生特征空间近邻平滑后的学生概率
        mgvaProb = knn_smooth(modelFeat, index_img, feat_bank, MB.model_bank, modelProb, args.knn_k) if args.knn_mgva else modelProb
        if args.t_ema:
            t_ema_bank[index_img] = args.t_ema_m * t_ema_bank[index_img] + (1 - args.t_ema_m) * modelProb
            mgvaProb = t_ema_bank[index_img]
        # --consensus teacher / both：老师学的不再是学生的 top-3，而是老师与学生（更新前）按熵加权的专家乘积
        #   第 1 步：学生终点 ≈ 训练前并集 + 0.5，「谁教谁」才是问题；COSMO 消融里共享共识比互相监督 +1.0
        # --ct_until：老师侧共识只在进度 < 该值时使用，之后老师改回学学生 top-k（原方法）。曲线（archive/2026_10_28_q1028）：共识在第 2–4 次评估
        #   所有任务都 +0.7 ~ +2.6（老师更准的早期），后期在学生领先的 →Cl / →Pr 上转负（Ar2Cl −2.7）；交叉点约在进度 7.5% ~ 12.5%（2026_10_17_collapse）
        if args.consensus in ("teacher", "both") and iter_num < int(args.ct_until * max_iter):
            base_mgva = mgvaProb
            mgvaProb = entropy_poe(clipProb, modelProb, *poe_ref)
            # --ct_marg：类级边际分歧门控。不判断样本上谁更准（两条谱系不可辨识），只看两边对「每类有多少张」的估计：记忆库上老师 / 学生 argmax
            #   张数相差超过 --ct_marg_thr 倍的类不做融合，老师学原目标（学生 top-k）。诊断（tools/diag_marg.py）：→Cl 上标出的正是被吞的类
            #   （Ar2Cl Computer 第 1 次评估 老师 6 / 学生 27 / 真实 99），→Ar 上几乎不触发（保留共识收益）
            if args.ct_marg:
                with torch.no_grad():
                    ct = torch.bincount(MB.vil_bank.argmax(dim=1), minlength=args.class_num).float()
                    cs = torch.bincount(MB.model_bank.argmax(dim=1), minlength=args.class_num).float()
                    flag = torch.log((ct + 1) / (cs + 1)).abs() > math.log(args.ct_marg_thr)
                    sel = flag[clipProb.argmax(dim=1)] | flag[modelProb.argmax(dim=1)]
                    if sel.any():
                        mgvaProb = torch.where(sel.unsqueeze(1), base_mgva, mgvaProb)
            # --ct_bal：老师学的共识也做学生侧同样的类均衡。诊断（archive/2026_10_28_q1028）：Clipart 上零样本 CLIP 一开始把 Computer 判成 Monitor
            #   （老师预测 10 / 真实 99，Monitor 446），不加共识时学生侧 1/频率均衡把 Computer 拉回、老师跟着学生恢复；加共识后老师部分在学自己，
            #   老师侧没有类均衡，初始偏差锁死，整类被吞（Computer 107 → 0）。这里把共识在全部样本上的类边际 m（记忆库上的同一个专家乘积）
            #   矫正到学生侧的目标分布 r：inv / self 冻结前 = 均匀，self 冻结后 / prior / oracle = clsw_prior；q ∝ 共识 · r / m
            if args.ct_bal:
                with torch.no_grad():
                    m = torch.clamp(entropy_poe(MB.vil_bank, MB.model_bank, *poe_ref).mean(dim=0), min=1e-4)
                    use_prior = args.clsw_mode in ("prior", "oracle") or (args.clsw_mode == "self" and iter_num >= int(args.clsw_self_at * max_iter))
                    r = clsw_prior if use_prior else torch.full_like(m, 1.0 / args.class_num)
                    mgvaProb = mgvaProb * (r / m).unsqueeze(0)
                    mgvaProb = mgvaProb / mgvaProb.sum(dim=1, keepdim=True)
        if split_fix is not None and args.split_teacher and sm.any():      # 老师也学改名后的类
            mgvaProb = mgvaProb.clone()
            mgvaProb[sm] = F.one_hot(fb[sm], args.class_num).float()
        top3_probs, top3_indices = torch.topk(mgvaProb, k=args.topk, dim=1)
        # --consensus tcand：候选类仍是学生的 top-3（保留学生纠正老师的通道），候选之间的权重换成共识
        #   分解（2026-10-24）：共识在 DN 上的增益全来自老师侧，OH 上老师侧共识 −1.6（学生后期更准，老师需要学生的候选）
        if args.consensus == "tcand":
            top3_indices = torch.topk(modelProb, k=args.topk, dim=1).indices
            top3_probs = entropy_poe(clipProb, modelProb, *poe_ref).gather(1, top3_indices)
        soft_labels = torch.zeros_like(logits_clip)
        soft_labels.scatter_(1, top3_indices, top3_probs)
        soft_labels = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-6)
        soft_labels = torch.softmax(soft_labels * 10, dim=1)

        probs_vil = torch.softmax(logits_clip, dim=1)
        log_vil = torch.log(probs_vil + 1e-6)
        log_soft_labels = torch.log(soft_labels + 1e-6)
        kl_elementwise = log_soft_labels - log_vil
        kl_div = torch.sum(soft_labels * kl_elementwise, dim=1)

        # --mgva_gate：MGVA 只在学生标签可靠的样本上教老师（第 1 步：adapter 学每个样本的学生 top-3，第 69 步老师 87.1 → 84.5、
        #   结束时一致率 99%，记住了学生的错误；只学提示的老师用同样的目标不掉、不塌缩）；其余样本交给 adapter 通过 CLIP 特征自己推广
        #   agree = 学生 top-1 和冻结的零样本 CLIP 一致；conf = 学生置信度在其预测类（按学生记忆库）里排前 --mgva_keep 的样本
        with torch.no_grad():
            if args.mgva_gate == "agree":
                mgva_mask = (modelProb.argmax(dim=1) == zs_pred[index_img]).float()
            elif args.mgva_gate == "conf":
                bank_conf, bank_pred = MB.model_bank.max(dim=1)
                thr = torch.ones(args.class_num, device=args.device)      # 记忆库里没有样本的类：该类的批内样本都不保留
                for c in bank_pred.unique():
                    thr[c] = torch.quantile(bank_conf[bank_pred == c], 1 - args.mgva_keep)
                conf_b, pred_b = modelProb.max(dim=1)
                mgva_mask = (conf_b >= thr[pred_b]).float()
            else:
                mgva_mask = torch.ones_like(kl_div)
        n_keep = mgva_mask.sum().clamp(min=1)                        # 只在保留的样本上取平均：损失量级和原方法一致

        loss_vil = (kl_div * mgva_mask).sum() / n_keep
        if args.t_strong:
            clipFeatS = CLIP_model.encode_image((strong_img * in_std + in_mean - clip_mean) / clip_std)
            clipFeatS = clipFeatS / clipFeatS.norm(dim=-1, keepdim=True)
            log_vil_s = torch.log(torch.softmax(100.0 * clipFeatS @ clipTextFeatGrad.T, dim=1) + 1e-6)
            loss_vil = (loss_vil + (torch.sum(soft_labels * (log_soft_labels - log_vil_s), dim=1) * mgva_mask).sum() / n_keep) / 2
        if args.ct_prompt:
            with torch.no_grad():
                cons = entropy_poe(clipProb, modelProb, *poe_ref)
                c_probs, c_idx = torch.topk(cons, k=args.topk, dim=1)        # 与 MGVA 目标同样的 top-k → 归一化 → softmax(10·) 处理
                cons_soft = torch.zeros_like(cons).scatter_(1, c_idx, c_probs)
                cons_soft = torch.softmax(cons_soft / (cons_soft.sum(dim=1, keepdim=True) + 1e-6) * 10, dim=1)
                log_cons = torch.log(cons_soft + 1e-6)
            views = [clipFeatWGrad.detach()] + ([clipFeatS.detach()] if args.t_strong else [])
            loss_pc = sum(torch.sum(cons_soft * (log_cons - torch.log_softmax(100.0 * v @ ctxTextFeat.T, dim=1)), dim=1).mean() for v in views) / len(views)
            loss_vil = loss_vil + loss_pc
        if args.t_im > 0:
            loss_vil = loss_vil + args.t_im * IML_t(probs_vil)
        # --t_proto_reg：把原型拉向初始文本嵌入（1 − 余弦相似度的类平均），保留语义锚点、防止漂移（离线迭代修正会越修越偏）
        if args.t_proto:
            loss_vil = loss_vil + args.t_proto_reg * (1 - (F.normalize(t_proto, dim=1) * t_proto_init).sum(1)).mean()
        # --freeze_teacher_at：到该进度后老师不再更新（仍计算 loss_vil 用于记录）
        if iter_num < freeze_teacher_iter:
            if clip_optimizer is not None:
                clip_optimizer.zero_grad()
            if prompt_optimizer is not None:
                prompt_optimizer.zero_grad()
            # --vil_low / --vil_after：到该进度后 MGVA（学生 → 老师）的学习率乘以 --vil_low（1.0 = 原方法）
            #   老师用 Adam，缩放损失几乎不改变更新幅度，所以缩放学习率
            #   Art 目标诊断：cap02 之后剩下的塌缩是对称的，老师吸收学生错误（Cl→Ar 32 个）和学生吸收老师错误（35 个）一样多
            if iter_num == vil_low_iter and args.vil_low != 1.0 and clip_optimizer is not None:
                for g in clip_optimizer.param_groups:
                    g["lr"] *= args.vil_low
                Project.log(f"MGVA lr x {args.vil_low} at iter {iter_num}", is_print=False)
            loss_vil.backward()
            if clip_optimizer is not None:
                clip_optimizer.step()
            if prompt_optimizer is not None:
                prompt_optimizer.step()
        elif iter_num == freeze_teacher_iter:
            Project.log(f"Teacher frozen at iter {iter_num}", is_print=False)
            tqdm.write(f"Teacher frozen at iter {iter_num}")

        ################################################
        model_LR.step()
        MB.update(index_img, vil_prob=clipProb, model_prob=modelProb)
        if feat_bank is not None:
            feat_bank[index_img] = F.normalize(modelFeat, dim=1)

        ################################################
        if args.log_dynamics:
            with torch.no_grad():
                y_b = item["label"].to(args.device)
                for k, v in (("loss_align", loss_align), ("loss_ctr", loss_ctr), ("loss_iml", loss_iml), ("loss_vil", loss_vil)):
                    dyn_acc[k] = dyn_acc.get(k, 0.0) + v.item()
                dyn_acc["n"] = dyn_acc.get("n", 0) + 1
                dyn_acc["pl_gamma"] = dyn_acc.get("pl_gamma", 0) + int((gamma1.argmax(dim=0) == y_b).sum())
                dyn_acc["pl_align"] = dyn_acc.get("pl_align", 0) + int((alignProb.argmax(dim=1) == y_b).sum())
                dyn_acc["m"] = dyn_acc.get("m", 0) + len(y_b)
                # MGVA 保留的样本数 / 其中学生 top-1 判对的数；--s_fm 时这两列改记 FixMatch 保留的样本（此时 MGVA 不门控）
                keep_mask = fm_mask if args.s_fm > 0 else mgva_mask
                dyn_acc["keep"] = dyn_acc.get("keep", 0) + int(keep_mask.sum())
                dyn_acc["keep_ok"] = dyn_acc.get("keep_ok", 0) + int(((modelProb.argmax(dim=1) == y_b).float() * keep_mask).sum())

        log_str = f'Task: SFDA, Test Target, {args.source} -> {args.target}, Iter: {iter_num:4d}/{max_iter:4d}; Align: {loss_align.item():.4f}; CTR: {loss_ctr.item():.4f}; IML: {loss_iml.item():.4f}; VLM: {loss_vil.item():.4f};'
        Project.log(log_str, is_print=False)
        bar.set_postfix(align=f"{loss_align.item():.3f}", ctr=f"{loss_ctr.item():.3f}", iml=f"{loss_iml.item():.3f}",
                        vlm=f"{loss_vil.item():.3f}", best=f"{best_acc:.4f}", refresh=False)
        if iter_num != 0 and (iter_num % eval_gap == 0 or iter_num == max_iter - 1):
            model_cm, model_oa, model_aa, model_class_acc = modeleval(args, feature_net, classifier_net, target_dataloader["test"])
            ModellineAcc = lineClassAcc(args, model_class_acc, iter_num, header=False)
            Project.log(ModellineAcc, filename=f"ModellineAcc.txt", is_print=False, haveTime=False)
            Project.log(f"\n\n\n ==>iter: {iter_num}  " + modeleval_str(model_oa,model_aa,model_cm, args.class_info), filename=f"Detail.txt", is_print = False)
            acc_total = model_oa
            if acc_total >= best_acc:
                best_acc = acc_total
            log_str = f'Task: SFDA, Test Target, {args.source} -> {args.target}, Iter: {iter_num:4d}/{max_iter:4d}; Now Acc: {acc_total:.4f}; Best Acc: {best_acc:.4f}'
            Project.log(log_str, is_print=False)
            tqdm.write(f"[eval] iter {iter_num:4d}/{max_iter}  acc {acc_total:.4f}  best {best_acc:.4f}")
            if ema_nets is not None:
                rng_state = get_rng_state()
                with torch.no_grad():
                    _, ema_oa, ema_aa, ema_class_acc = modeleval(args, ema_nets[0], ema_nets[1], target_dataloader["test"])
                set_rng_state(rng_state)
                Project.log(lineClassAcc(args, ema_class_acc, iter_num, header=False), filename="ModellineAcc_ema.txt", is_print=False, haveTime=False)
                Project.log(f"[ema] iter {iter_num}  OA {ema_oa:.4f}  AA {ema_aa:.4f}（原 OA {model_oa:.4f}  AA {model_aa:.4f}）", is_print=False)
                tqdm.write(f"[ema] iter {iter_num:4d}/{max_iter}  acc {ema_oa:.4f}  aa {ema_aa:.4f}")
            if args.eval_bn_recal:
                _, bn_oa, bn_aa, bn_class_acc = bn_recal_eval(args, feature_net, classifier_net, target_dataloader["test"])
                Project.log(lineClassAcc(args, bn_class_acc, iter_num, header=False), filename="ModellineAcc_bnrecal.txt", is_print=False, haveTime=False)
                Project.log(f"[bn recal] iter {iter_num}  OA {bn_oa:.4f}  AA {bn_aa:.4f}（原 OA {model_oa:.4f}  AA {model_aa:.4f}）", is_print=False)
            if args.log_dynamics:
                cur = eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, target_dataloader["test"], args.device, text_fn)
                if args.log_samples:
                    save_samples(samples_path, samples_hist, iter_num, cur)
                if args.log_rich:
                    rich_n += 1      # 特征每 4 次评估存一次
                    save_rich(Project.root_path, rich_hist, iter_num, eval_rich(feature_net, classifier_net, CLIP_model, cls_text,
                              target_dataloader["test"], args.device, text_fn, with_feat=rich_n % 4 == 0), rich_label)
                r = log_dynamics(dyn_path, iter_num, now_epoch, cur, dyn_prev, dyn_acc, model_optimizer.param_groups[1]["lr"])
                tqdm.write(f"[dyn] iter {iter_num:4d}  老师 {r['teacher_acc']:.4f}  一致率 {r['agree']:.3f}  两边都错 {r['both_wrong']}  "
                           f"翻转 学生 {r['student_flip']:.3f} / 老师 {r['teacher_flip']:.3f}  伪标签 {r['pl_gamma_acc']:.4f}")
                dyn_prev, dyn_acc = cur, {}
        iter_num += 1
        bar.update(1)
    bar.close()

    # 训练结束时的完整模型（汇报数字对应的那一份；--s_ema > 0 时另存 EMA 学生），只写文件不影响训练
    if args.save_final:
        torch.save({
            "iter": iter_num, "max_iter": max_iter,
            "args": vars(args),
            "imgs": target_dataloader["test"].dataset.imgs,
            "feature": feature_net.state_dict(),
            "classifier": classifier_net.state_dict(),
            "ema_feature": ema_nets[0].state_dict() if ema_nets is not None else None,
            "ema_classifier": ema_nets[1].state_dict() if ema_nets is not None else None,
            "adapter": {k: v for k, v in CLIP_model.state_dict().items() if "adapter" in k},
            "prompt_ctx": prompt_learner.ctx.detach().cpu() if prompt_learner is not None else None,
            "vil_bank": MB.vil_bank.cpu(),
            "model_bank": MB.model_bank.cpu(),
        }, os.path.join(Project.root_path, "final.pt"))
        Project.log(f"Final checkpoint saved (iter {iter_num})", is_print=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=None, help="DataLoader 进程数；默认沿用原设置（4 / 5）。VisDA / DomainNet 图片大，数据加载是瓶颈时调大")
    parser.add_argument('--data_frac', type=float, default=1.0, help="目标域每类只取这个比例（固定子集，训练和测试都用它），用于快速筛选；1.0 = 全量")
    parser.add_argument('--source', type=str, default="webcam")
    parser.add_argument('--target', type=str, default="amazon")
    parser.add_argument('--dataset', type=str, default="OFFICE31")
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument("--mbMethod", default="aware", type=str, choices=["fix", "linear", "aware"])
    parser.add_argument("--fixAlpha", default=0.8, type=float)
    parser.add_argument('--clipVersion', type=str, default="ViT-B/32", choices=['ViT-B/32','ViT-B/16','ViT-B/14'])

    parser.add_argument("--lr", default=1e-2, type=float, metavar="model_LR", help="initial learning rate", dest="lr")
    parser.add_argument('--adaptercount', type=int, default=3, choices=[1,2,3,4,5,6,7,8,9,10,11,12])

    parser.add_argument('--otReg', type=float, default=0.2)

    parser.add_argument("--weightalign", default=1, type=float)
    parser.add_argument("--weightvil", default=1, type=float)
    parser.add_argument("--weightctr", default=1, type=float)
    parser.add_argument("--weightiml", default=1, type=float)
    parser.add_argument("--imloss", default=1, type=float)

    parser.add_argument("--topk", default=3, type=int)

    parser.add_argument("--snapshot_at", default=[], type=float, nargs="*", help="保存快照的训练进度，如 0 0.25 0.5")
    parser.add_argument("--src_suffix", default="", type=str, help="源模型目录后缀，如 _s0 → runs/Source_home_s0/；默认空 = seed 42 的源模型")
    parser.add_argument("--save_final", default=1, type=int, help="1 = 训练结束时保存 final.pt（学生 + EMA 学生 + adapter + 记忆库）")
    parser.add_argument("--stop_at", default=1.0, type=float, help="训练到该进度后停止，如 0.5")
    parser.add_argument("--align_mode", default="ot", type=str, choices=["ot", "clip_soft", "clip_hard", "clip_sharp"],
                        help="对齐损失用的计划：ot = 原方法；clip_soft / clip_hard = 消融，直接用 CLIP 软 / 硬标签；"
                             "clip_sharp = 消融，和 OT 同一个核但不做类别边际投影")
    parser.add_argument("--knn_k", default=0, type=int,
                        help="> 0 时，对齐损失用的 CLIP 概率换成学生特征空间 k 近邻平滑后的版本：(自身 + 近邻平均) / 2；0 = 关闭（原方法）")
    parser.add_argument("--align_view", default="weak", type=str, choices=["weak", "strong", "both"],
                        help="学生在哪个视图上对齐老师的计划：weak = 原方法；strong = 强增强视图（Noisy Student）；both = 两者平均")
    parser.add_argument("--log_dynamics", action="store_true",
                        help="每个 epoch 记录训练动态到 dynamics.csv（老师准确率、一致率、翻转率、熵、伪标签准确率、各项损失）；不影响训练过程")
    parser.add_argument("--knn_mgva", action="store_true",
                        help="MGVA 中老师学的学生概率也做同样的近邻平滑（需要 --knn_k > 0）")

    parser.add_argument("--log_samples", action="store_true",
                        help="和 --log_dynamics 一起用：每次记录时另存逐样本的学生 / 老师预测和熵到 samples.npz；不影响训练过程")
    parser.add_argument("--log_rich", action="store_true",
                        help="和 --log_dynamics 一起用：每次记录时另存学生 / 老师的完整概率（含水平翻转）到 rich.npz，每 4 次另存两边特征；不影响训练过程")
    parser.add_argument("--freeze_teacher_at", default=1.0, type=float,
                        help="训练到该进度后不再更新老师（CLIP adapter），如 0.2；1.0 = 不冻结（原方法）")
    parser.add_argument("--knn_self_after", default=1.0, type=float,
                        help="训练到该进度后，近邻平滑的概率改为 CLIP 与学生的加权平均（需要 --knn_k > 0）；1.0 = 不启用")
    parser.add_argument("--knn_self_w", default=0.5, type=float, help="--knn_self_after 启用后学生概率的权重")
    parser.add_argument("--align_agree_after", default=1.0, type=float,
                        help="训练到该进度后，只在学生预测 = 计划 argmax 的样本上做对齐；1.0 = 不启用（原方法）")
    parser.add_argument("--align_weight", default="ramp", type=str, choices=["ramp", "const", "cap"],
                        help="对齐权重的进度曲线：ramp = min(1, 2·进度)（原方法）；const = 1；cap = min(--align_cap, 2·进度)")
    parser.add_argument("--align_cap", default=0.2, type=float, help="--align_weight cap 的上限（0.2 = 进度 0.1 时达到）")
    parser.add_argument("--align_norm_c", default=0, type=float, help="对齐权重乘以 该值 / 类别数（按类别数归一化对齐损失量级；65 = 以 Office-Home 为参考，OH 不变）；0 = 关闭")
    parser.add_argument("--vil_after", default=1.0, type=float, help="到该训练进度后 MGVA（老师 adapter）的学习率乘以 --vil_low；1.0 = 不启用（原方法）")
    parser.add_argument("--vil_low", default=1.0, type=float, help="--vil_after 之后 MGVA 学习率的倍数（0 = 等价于冻结老师）")
    parser.add_argument("--src_expert", action="store_true",
                        help="对齐目标中的 CLIP 概率换成 CLIP 与冻结源模型按熵加权的专家乘积（默认关闭）")
    parser.add_argument("--src_expert_until", default=1.0, type=float, help="--src_expert 使用到该训练进度为止；1.0 = 全程")
    # 组件审计（2026-10-19，scripts/run_audit.sh）；diversity 项用已有的 --imloss 0，SimSiam 用 --weightctr 0，MGVA 用 --freeze_teacher_at 0
    parser.add_argument("--no_clsw", action="store_true", help="消融：对齐计划不再乘以学生预测频率的倒数（默认关闭 = 原方法）")
    parser.add_argument("--iml_ent", default=1.0, type=float, help="IM 中样本熵项的权重；1.0 = 原方法，0 = 去掉")
    parser.add_argument("--cls_lr_mult", default=0.1, type=float, help="分类头学习率相对 --lr 的倍数；0.1 = 原方法")
    parser.add_argument("--s_ema", default=0.0, type=float, help="学生权重 + BN 统计量的 EMA，时间常数（epoch 数）；只额外评估、记录 ModellineAcc_ema.txt，不影响训练；0 = 关闭")
    parser.add_argument("--eval_bn_recal", action="store_true", help="诊断：每次评估另外记录「BN 统计量在目标域上重估后」的准确率（ModellineAcc_bnrecal.txt），不影响训练和原指标")
    parser.add_argument("--clsw_self_em", action="store_true", help="--clsw_mode self 冻结时用标签偏移 EM（MLLS，训练先验视为均匀）重估目标分布，而不是直接取平均概率")
    parser.add_argument("--clsw_mode", default="inv", type=str, choices=["inv", "const", "sqrt", "prior", "pow", "oracle", "self"],
                        help="对齐计划的类频率重加权：inv = 1/频率（原方法）；const = 不均衡；sqrt = 1/√频率；prior = CLIP 零样本平均概率 / 频率；oracle = 真实类频率 / 频率（诊断上界，用了标签）；self = 进度 --clsw_self_at 时冻结的学生平均概率 / 频率（之前同 inv）；"
                             "后三者每步把损失量级调成与 inv 相同")
    # 方向 E（2026-10-22）：Ar→Pr 的剩余错误是成对的语义混淆（Trash_Can→Bucket 19、Folder→Notebook 19、Marker→Pen 18…），
    #   被吞并的类预测数只有真实的 0.69 ~ 0.74；sqrt（更弱的均衡）在 Product 上 −1.5，试更强的均衡
    # 混淆类拆分纠正（2026-10-22，scripts/run_splitfix.sh）：tools/diag_split.py 早期快照上无标签改名为正
    parser.add_argument("--split_at", default=1.0, type=float, help="到该训练进度做混淆类拆分纠正；1.0 = 不做")
    parser.add_argument("--split_until", default=0.0, type=float, help="> --split_at 时，之后每个 epoch 重做一次直到该进度")
    parser.add_argument("--split_k", default=3, type=int, help="每个预测类拆成的子簇数")
    parser.add_argument("--split_theta", default=0.2, type=float, help="子簇平均概率 第二名 / 预测类 的阈值")
    parser.add_argument("--split_release", default=1.0, type=float, help="到该训练进度撤掉改名（需 > --split_until）；1.0 = 一直用")
    parser.add_argument("--split_teacher", action="store_true", help="老师（MGVA）也学改名后的类")
    parser.add_argument("--t_flip", action="store_true", help="老师给学生的预测取原图与水平翻转图的概率平均（默认关闭）")
    parser.add_argument("--clsw_self_at", default=0.25, type=float, help="--clsw_mode self：冻结目标分布的训练进度")
    parser.add_argument("--clsw_pow", default=1.0, type=float, help="--clsw_mode pow：权重 = 1 / 频率^p（量级同 inv）；1.0 = inv，0.5 = sqrt")
    parser.add_argument("--t_prompt", action="store_true", help="老师文本端加可学习的共享提示上下文（默认关闭 = 固定提示 'a photo of a'）")
    parser.add_argument("--t_prompt_nctx", default=4, type=int, help="--t_prompt 的上下文向量个数（用提示词前几个词初始化）")
    parser.add_argument("--t_prompt_lr", default=1e-5, type=float, help="--t_prompt 上下文的 Adam 学习率（adapter 为 5e-5）")
    parser.add_argument("--t_prompt_opt", default="adam", choices=["adam", "sgd"],
                        help="提示上下文的优化器：adam = 和 adapter 同一个 Adam（--t_prompt_lr）；sgd = DIFO 系列公开代码的设置（SGD 0.9，wd 5e-4，lr = --lr × 0.1）")
    # 方向 A（2026-10-21，scripts/run_gate.sh）：MGVA 只在学生可靠的样本上教老师
    parser.add_argument("--mgva_gate", default="none", choices=["none", "agree", "conf"],
                        help="none = 原方法（所有样本）；agree = 学生和冻结零样本 CLIP 一致；conf = 学生置信度在预测类内排前 --mgva_keep")
    parser.add_argument("--mgva_keep", default=0.5, type=float, help="--mgva_gate conf 时每个预测类保留的比例")
    parser.add_argument("--no_adapter", action="store_true", help="和 --t_prompt 一起用：图像侧 adapter 不更新，只学提示上下文")
    parser.add_argument("--t_proto", action="store_true", help="老师的类原型变成可学习参数（文本嵌入初始化；默认关闭 = 固定文本原型）")
    parser.add_argument("--t_proto_reg", default=1.0, type=float, help="--t_proto 时把原型拉向初始文本嵌入的权重")
    parser.add_argument("--t_ema", action="store_true", help="老师（MGVA）学的学生概率换成逐样本动量平均（默认关闭 = 原方法）")
    parser.add_argument("--t_ema_m", default=0.7, type=float, help="--t_ema 的动量（每个样本每个 epoch 更新一次；0.7 ≈ 记住约 3 个 epoch）")
    parser.add_argument("--t_strong", action="store_true", help="老师也在强增强视图上学同一个目标（默认关闭）")
    parser.add_argument("--t_im", default=0.0, type=float, help="老师损失加上自身预测 IM 的权重；0 = 原方法")
    # 方向 B（2026-10-22，scripts/run_cons.sh）：COSMO 式共享共识
    parser.add_argument("--consensus", default="none", choices=["none", "teacher", "student", "both", "tcand"],
                        help="none = 原方法（老师学学生 top-3，学生学老师）；teacher = 老师学「老师 × 学生」熵加权专家乘积；student = 只有学生对齐目标换成它；both = 两边都换；tcand = 老师的候选类仍取学生 top-3，候选间权重用共识")
    parser.add_argument("--ct_bal", action="store_true",
                        help="和 --consensus teacher / both 一起用：老师学的共识按记忆库上的类边际矫正到学生侧类均衡的目标分布（均匀或 clsw_prior）")
    parser.add_argument("--ct_marg", action="store_true",
                        help="和 --consensus teacher / both 一起用：老师 / 学生记忆库上预测张数相差超过 --ct_marg_thr 倍的类不做融合，老师学原目标")
    parser.add_argument("--ct_marg_thr", default=3.0, type=float, help="--ct_marg 的张数比阈值")
    parser.add_argument("--ct_prompt", action="store_true",
                        help="和 --t_prompt 一起用：adapter 学学生 top-k（原方法），提示上下文学老师与学生的专家乘积共识（两路梯度分开）")
    parser.add_argument("--ct_until", default=1.0, type=float,
                        help="老师侧共识（--consensus teacher / both）只在训练进度 < 该值时使用，之后老师学学生 top-k；1.0 = 全程")
    parser.add_argument("--poe_weight", default="ent", choices=["ent", "rank"],
                        help="共识的专家权重：ent = 绝对熵（COSMO）；rank = 熵在各自模型全部样本中的百分位（消掉两个模型的校准差别）")
    parser.add_argument("--s_ce", default=0.0, type=float, help="学生对共识 argmax 的硬交叉熵权重；0 = 不用")
    # 方向 C（2026-10-22，scripts/run_self.sh）：学生自训练
    parser.add_argument("--s_fm", default=0.0, type=float, help="FixMatch 损失权重（弱视图自身高置信伪标签 → 强视图交叉熵）；0 = 不用")
    parser.add_argument("--fm_tau", default=0.95, type=float, help="--s_fm 的置信度阈值")
    parser.add_argument("--s_mix", default=0.0, type=float, help="批内 mixup 损失权重（目标为学生 argmax 的混合）；0 = 不用")
    parser.add_argument("--mix_alpha", default=0.3, type=float, help="--s_mix 的 Beta 分布参数")
    parser.add_argument("--s_aad", default=0.0, type=float, help="AaD 近邻吸引 / 批内分散损失的权重（近邻数 = --knn_k）；0 = 不用")

    args = parser.parse_args()
    assert not args.knn_mgva or args.knn_k > 0, "--knn_mgva 需要 --knn_k > 0"
    assert not args.log_samples or args.log_dynamics, "--log_samples 需要 --log_dynamics"
    assert not args.log_rich or args.log_dynamics, "--log_rich 需要 --log_dynamics"
    assert not args.ct_bal or args.consensus in ("teacher", "both"), "--ct_bal 需要 --consensus teacher / both"
    assert not args.ct_marg or args.consensus in ("teacher", "both"), "--ct_marg 需要 --consensus teacher / both"
    assert args.knn_self_after >= 1.0 or args.knn_k > 0, "--knn_self_after 需要 --knn_k > 0"
    assert not args.src_expert or args.knn_self_after >= 1.0, "--src_expert 不和 --knn_self_after 同时使用"
    assert not (args.t_ema and args.knn_mgva), "--t_ema 不和 --knn_mgva 同时使用"
    assert not (args.t_prompt and args.t_proto), "--t_prompt 不和 --t_proto 同时使用"
    assert not args.no_adapter or args.t_prompt, "--no_adapter 需要 --t_prompt"
    assert not args.ct_prompt or (args.t_prompt and not args.no_adapter and args.consensus == "none"), "--ct_prompt 需要 --t_prompt，不和 --no_adapter / --consensus 同时使用"
    assert not (args.consensus == "both" and args.src_expert), "--consensus both 不和 --src_expert 同时使用"
    assert not args.s_aad > 0 or args.knn_k > 0, "--s_aad 需要 --knn_k > 0"
    assert not (args.s_fm > 0 and args.mgva_gate != "none"), "--s_fm 借用 MGVA 保留的记录列，不和 --mgva_gate 同时使用"
    assert not (args.no_clsw and args.clsw_mode != "inv"), "--no_clsw 不和 --clsw_mode 同时使用"

    seed_everything(args.seed)

    if args.dataset == "OFFICE31":
        args.class_num = 31
        args.domains = ["amazon", "dslr", "webcam"]
        args.class_info = ['back_pack','bike','bike_helmet','bookcase','bottle','calculator','desktop_computer','desk_chair','desk_lamp','file_cabinet','headphones','keyboard','laptop_computer','letter_tray','mobile_phone','monitor','mouse','mug','paper_notebook','pen','phone','printer','projector','punchers','ring_binder','ruler','scissors','speaker','stapler','tape_dispenser','trash_can']
        args.sourceStorage = "Source_31"
    elif args.dataset == "OfficeHomeDataset":
        args.class_num = 65
        args.source = {"art":"Art","clipart":"Clipart","product":"Product","realworld":"Real World"}[args.source]  # 转义「Real World」因空格无法输入命令行的问题
        args.target = {"art":"Art","clipart":"Clipart","product":"Product","realworld":"Real World"}[args.target]
        args.domains = ["Art", "Clipart", "Product", "Real World"]
        # 类名从 <image_root>/classname.txt 读取，每行一个，行号即标签编号（原为写死的列表，和云端列表的标签顺序不一致）
        args.class_info = load_class_names(os.path.join(IMAGEFILE, args.dataset, "classname.txt"), args.class_num)
        args.sourceStorage = "Source_home"
    elif args.dataset == "VisDA":
        args.class_num = 12
        args.domains = ["synthesis", "real"]
        args.class_info = ['aeroplane','bicycle','bus','car','horse','knife','motorcycle','person','plant','skateboard','train','truck']
        args.sourceStorage = "Source_visda"
    elif args.dataset == "DomainNet":
        args.class_num = 126
        args.domains = ["clipart", "painting", "real", "sketch"]
        args.class_info = ['aircraft_carrier', 'alarm_clock', 'ant', 'anvil', 'asparagus', 'axe', 'banana', 'basket', 'bathtub', 'bear', 'bee', 'bird', 'blackberry', 'blueberry', 'bottlecap', 'broccoli', 'bus', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon', 'canoe', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'coffee_cup', 'compass', 'computer', 'cow', 'crab', 'crocodile', 'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck', 'dumbbell', 'elephant', 'eyeglasses', 'feather', 'fence', 'fish', 'flamingo', 'flower', 'foot', 'fork', 'frog', 'giraffe', 'goatee', 'grapes', 'guitar', 'hammer', 'helicopter', 'helmet', 'horse', 'kangaroo', 'lantern', 'laptop', 'leaf', 'lion', 'lipstick', 'lobster', 'microphone', 'monkey', 'mosquito', 'mouse', 'mug', 'mushroom', 'onion', 'panda', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'pig', 'pillow', 'pineapple', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone', 'screwdriver', 'sea_turtle', 'see_saw', 'sheep', 'shoe', 'skateboard', 'snake', 'speedboat', 'spider', 'squirrel', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'swan', 'table', 'teapot', 'teddy-bear', 'television', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'tiger', 'toe', 'train', 'truck', 'umbrella', 'vase', 'watermelon', 'whale', 'zebra']
        args.sourceStorage = "Source_domainnet"

    args.sourceStorage += args.src_suffix
    assert args.source in args.domains, f"error dataset or domai: {args.source} not in {args.dataset}"
    args.image_root = os.path.join(IMAGEFILE, args.dataset)
    args.device = f"cuda:{args.device}"
    # 核对类名顺序和源域 / 目标域图片列表的标签一致，否则 CLIP 提示词会配错类别
    if args.dataset == "OfficeHomeDataset":
        check_class_names(args.class_info, [os.path.join("dataset", args.dataset, f"{d}.txt") for d in (args.source, args.target)])

    filename = os.path.basename(sys.argv[0])[:-3] + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if args.name == None:
        Project(project_name=filename, root_name="runs")    
    else:
        Project(project_name=args.name, root_name="runs")
    args_str = json.dumps(vars(args), ensure_ascii=False, indent=2)
    Project.log(args_str, filename=f"argsInfo.txt", is_print = False)
    Project.log(sys.argv[0])
    Project.log(f"Task: SFDA, Source:{args.source}, Target:{args.target}")
    backup_files_and_folders(Project.root_path, sys.argv[0])

    # 训练计时：开始 / 结束 / 总耗时（含加载 CLIP、初始化记忆库），写入 run_log.txt
    t_begin = datetime.datetime.now()
    Project.log(f"【训练开始】{t_begin.strftime('%Y-%m-%d %H:%M:%S')}", haveTime=False)

    CLIP_model, CLIP_preprocess = clip.load(args.clipVersion, device=args.device)
    target_dataloader = getDataloaderSFDA5X(args, args.target, CLIP_preprocess, True)
    start(args, target_dataloader, CLIP_model)

    t_end = datetime.datetime.now()
    Project.log(f"【训练结束】{t_end.strftime('%Y-%m-%d %H:%M:%S')}", haveTime=False)
    Project.log(f"【总耗时】{datetime.timedelta(seconds=int((t_end - t_begin).total_seconds()))}", haveTime=False)
