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
from model.simsiam import SimSiam, fastSimsiam
import datetime
import json
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np


IMAGEFILE = "dataset"    # 相对项目根目录；图片通过软链接放在 dataset/<数据集>/<域>/ 下

def get_rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def set_rng_state(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


def eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, loader, device):
    """训练动态：学生 / 老师（带 adapter 的 CLIP）在目标域全部图片上的预测、熵和标签（标签只用于记录）"""
    rng_state = get_rng_state()          # 遍历 DataLoader 会消耗随机数，恢复后训练过程和不记录时完全一致
    feature_net.eval(); CLIP_model.eval()
    N = len(loader.dataset)
    out = {k: torch.zeros(N, dtype=torch.long, device=device) for k in ("s_pred", "t_pred", "label")}
    out.update({k: torch.zeros(N, device=device) for k in ("s_ent", "t_ent")})
    with torch.no_grad():
        tf = CLIP_model.encode_text(cls_text)
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


def entropy_poe(p, q):
    """按熵加权的专家乘积（COSMO / ProDe 的源模型去噪思路）：r ∝ p^a · q^(1-a)，a = e^{-H(p)} / (e^{-H(p)} + e^{-H(q)})，熵越小权重越大"""
    hp = -(p * torch.log(p + 1e-8)).sum(1, keepdim=True)
    hq = -(q * torch.log(q + 1e-8)).sum(1, keepdim=True)
    a = torch.exp(-hp) / (torch.exp(-hp) + torch.exp(-hq))
    return torch.softmax(a * torch.log(p + 1e-8) + (1 - a) * torch.log(q + 1e-8), dim=1)


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

    MB = MemoryBank(target_dataloader["test"].dataset.__len__(), args.class_num, device=args.device)
    with torch.no_grad():
        cls_text_features = CLIP_model_raw.encode_text(cls_text)
        MB.initialize(target_dataloader["test"], CLIP_model_raw, feature_net, classifier_net, cls_text_features, args.device)
    # --knn_k > 0：在学生特征空间里用近邻平滑伪标签（静态诊断见 archive/2026_10_16_diag_struct）
    feat_bank = init_feat_bank(feature_net, target_dataloader["test"], args.device) if args.knn_k > 0 else None
    # --src_expert：冻结源模型的概率库（必须在训练开始前、学生还等于源模型时计算）
    src_bank = init_src_bank(feature_net, classifier_net, target_dataloader["test"], args.device) if args.src_expert else None
    SSNB = SimSiam(feature_net).to(args.device)

    model_cm, model_oa, model_aa, model_class_acc = modeleval(args, feature_net, classifier_net, target_dataloader["test"])
    ModellineAcc = lineClassAcc(args, model_class_acc, 0, header=True)
    Project.log(ModellineAcc, filename=f"ModellineAcc.txt", is_print=False, haveTime=False)
    Project.log(f"\n\n\n ==>iter: {iter_num}  " + modeleval_str(model_oa,model_aa,model_cm, args.class_info), filename=f"Detail.txt", is_print = False)
    best_acc = model_oa

    IML = InformationMaximizationLoss(trade_off=args.imloss, ent_weight=args.iml_ent)

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
        dyn_prev = eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, target_dataloader["test"], args.device)
        if args.log_samples:
            save_samples(samples_path, samples_hist, 0, dyn_prev)
        r = log_dynamics(dyn_path, 0, 0, dyn_prev, None, {}, model_optimizer.param_groups[1]["lr"])
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
                "vil_bank": MB.vil_bank.cpu(),
                "model_bank": MB.model_bank.cpu(),
            }, os.path.join(Project.root_path, f"snapshot_p{round(p * 100):03d}.pt"))
            Project.log(f"Snapshot saved at progress {p} (iter {iter_num})", is_print=False)
            tqdm.write(f"Snapshot saved at progress {p} (iter {iter_num})")
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
            vilTextFeat = CLIP_model.encode_text(cls_text)      # [C, FV]
            vilImgFeat = vilImgFeat / vilImgFeat.norm(dim=-1, keepdim=True)         # [B, FV]
            vilTextFeat = vilTextFeat / vilTextFeat.norm(dim=-1, keepdim=True)      # [B, FC]
            clipSim = vilImgFeat @ vilTextFeat.T                                    # [B, C]
            clipProb = torch.softmax(100*clipSim, dim = 1)                          # [B, C]
            # 对齐损失用的 CLIP 概率：开启 --knn_k 时换成学生特征空间近邻平滑后的版本（记忆库仍存原始 clipProb）
            # --knn_self_after：到该进度后，平滑用的自身概率和近邻概率都换成 (1 - w) · CLIP + w · 学生（w = --knn_self_w），
            #   让对齐目标在后期仍高于学生自己（静态诊断 p050：学生自身的近邻平滑比学生高 +1.0）；μ_p 仍取 CLIP 记忆库
            # --src_expert：对齐目标里的 CLIP 概率（自身和近邻的）换成 CLIP 与冻结源模型按熵加权的专家乘积；
            #   诊断：p000 时 Ar→Cl 有 407 个样本只有源模型判对，原方法训练开始后不再使用源模型
            if args.src_expert and iter_num < src_expert_iter:
                clipAlign = entropy_poe(clipProb, src_bank[index_img])
                vilBankAlign = entropy_poe(MB.vil_bank, src_bank)
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

        loss1 = args.weightalign * weight_align * loss_align + \
                args.weightctr * loss_ctr + \
                args.weightiml * loss_iml

        model_optimizer.zero_grad()
        loss1.backward()
        model_optimizer.step()

        #########################################################

        clipFeatWGrad = CLIP_model.encode_image(clip_img)
        clipFeatWGrad = clipFeatWGrad / clipFeatWGrad.norm(dim=-1, keepdim=True)

        clipTextFeatGrad = CLIP_model.encode_text(cls_text)
        clipTextFeatGrad = clipTextFeatGrad / clipTextFeatGrad.norm(dim=-1, keepdim=True)

        logits_clip = (100.0 * clipFeatWGrad @ clipTextFeatGrad.T)

        # --knn_mgva：老师（CLIP adapter）学的目标换成学生特征空间近邻平滑后的学生概率
        mgvaProb = knn_smooth(modelFeat, index_img, feat_bank, MB.model_bank, modelProb, args.knn_k) if args.knn_mgva else modelProb
        top3_probs, top3_indices = torch.topk(mgvaProb, k=args.topk, dim=1)
        soft_labels = torch.zeros_like(logits_clip)
        soft_labels.scatter_(1, top3_indices, top3_probs)
        soft_labels = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-6)
        soft_labels = torch.softmax(soft_labels * 10, dim=1)

        probs_vil = torch.softmax(logits_clip, dim=1)
        log_vil = torch.log(probs_vil + 1e-6)
        log_soft_labels = torch.log(soft_labels + 1e-6)
        kl_elementwise = log_soft_labels - log_vil
        kl_div = torch.sum(soft_labels * kl_elementwise, dim=1)

        loss_vil = torch.mean(kl_div)
        # --freeze_teacher_at：到该进度后老师不再更新（仍计算 loss_vil 用于记录）
        if iter_num < freeze_teacher_iter:
            clip_optimizer.zero_grad()
            # --vil_low / --vil_after：到该进度后 MGVA（学生 → 老师）的学习率乘以 --vil_low（1.0 = 原方法）
            #   老师用 Adam，缩放损失几乎不改变更新幅度，所以缩放学习率
            #   Art 目标诊断：cap02 之后剩下的塌缩是对称的，老师吸收学生错误（Cl→Ar 32 个）和学生吸收老师错误（35 个）一样多
            if iter_num == vil_low_iter and args.vil_low != 1.0:
                for g in clip_optimizer.param_groups:
                    g["lr"] *= args.vil_low
                Project.log(f"MGVA lr x {args.vil_low} at iter {iter_num}", is_print=False)
            loss_vil.backward()
            clip_optimizer.step()
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
            if args.log_dynamics:
                cur = eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, target_dataloader["test"], args.device)
                if args.log_samples:
                    save_samples(samples_path, samples_hist, iter_num, cur)
                r = log_dynamics(dyn_path, iter_num, now_epoch, cur, dyn_prev, dyn_acc, model_optimizer.param_groups[1]["lr"])
                tqdm.write(f"[dyn] iter {iter_num:4d}  老师 {r['teacher_acc']:.4f}  一致率 {r['agree']:.3f}  两边都错 {r['both_wrong']}  "
                           f"翻转 学生 {r['student_flip']:.3f} / 老师 {r['teacher_flip']:.3f}  伪标签 {r['pl_gamma_acc']:.4f}")
                dyn_prev, dyn_acc = cur, {}
        iter_num += 1
        bar.update(1)
    bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
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
    parser.add_argument("--vil_after", default=1.0, type=float, help="到该训练进度后 MGVA（老师 adapter）的学习率乘以 --vil_low；1.0 = 不启用（原方法）")
    parser.add_argument("--vil_low", default=1.0, type=float, help="--vil_after 之后 MGVA 学习率的倍数（0 = 等价于冻结老师）")
    parser.add_argument("--src_expert", action="store_true",
                        help="对齐目标中的 CLIP 概率换成 CLIP 与冻结源模型按熵加权的专家乘积（默认关闭）")
    parser.add_argument("--src_expert_until", default=1.0, type=float, help="--src_expert 使用到该训练进度为止；1.0 = 全程")
    # 组件审计（2026-10-19，scripts/run_audit.sh）；diversity 项用已有的 --imloss 0，SimSiam 用 --weightctr 0，MGVA 用 --freeze_teacher_at 0
    parser.add_argument("--no_clsw", action="store_true", help="消融：对齐计划不再乘以学生预测频率的倒数（默认关闭 = 原方法）")
    parser.add_argument("--iml_ent", default=1.0, type=float, help="IM 中样本熵项的权重；1.0 = 原方法，0 = 去掉")
    parser.add_argument("--cls_lr_mult", default=0.1, type=float, help="分类头学习率相对 --lr 的倍数；0.1 = 原方法")

    args = parser.parse_args()
    assert not args.knn_mgva or args.knn_k > 0, "--knn_mgva 需要 --knn_k > 0"
    assert not args.log_samples or args.log_dynamics, "--log_samples 需要 --log_dynamics"
    assert args.knn_self_after >= 1.0 or args.knn_k > 0, "--knn_self_after 需要 --knn_k > 0"
    assert not args.src_expert or args.knn_self_after >= 1.0, "--src_expert 不和 --knn_self_after 同时使用"

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
