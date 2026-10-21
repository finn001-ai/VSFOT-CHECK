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
            # 对齐损失用的 CLIP 概率：开启 --knn_k 时换成学生特征空间近邻平滑后的版本（记忆库仍存原始 clipProb）
            # --knn_self_after：到该进度后，平滑用的自身概率和近邻概率都换成 (1 - w) · CLIP + w · 学生（w = --knn_self_w），
            #   让对齐目标在后期仍高于学生自己（静态诊断 p050：学生自身的近邻平滑比学生高 +1.0）；μ_p 仍取 CLIP 记忆库
            # --src_expert：对齐目标里的 CLIP 概率（自身和近邻的）换成 CLIP 与冻结源模型按熵加权的专家乘积；
            #   诊断：p000 时 Ar→Cl 有 407 个样本只有源模型判对，原方法训练开始后不再使用源模型
            if args.src_expert and iter_num < src_expert_iter:
                clipAlign = entropy_poe(clipProb, src_bank[index_img])
                vilBankAlign = entropy_poe(MB.vil_bank, src_bank)
            # --consensus both：学生对齐的目标（自身和近邻的）换成老师与学生按熵加权的专家乘积（COSMO 式共享共识，更新前的预测，detach）
            elif args.consensus == "both":
                clipAlign = entropy_poe(clipProb, modelProb)
                vilBankAlign = entropy_poe(MB.vil_bank, MB.model_bank)
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
        # --clsw_mode：类频率重加权的形状（inv = 原方法 1/频率）。权重均值约为类别数，直接去掉会让对齐损失缩小约 65 倍（审计 noclsw 崩溃的原因），
        #   所以 const / sqrt 每一步都把「按计划质量加权的平均权重」调成和 inv 相同：损失量级不变，只改变类别之间的相对权重
        #   const = 所有类同一权重（不做均衡）；sqrt = 1/√频率（均衡推得轻一半）
        #   prior = CLIP 零样本平均概率 / 频率：把学生的预测分布推向 CLIP 估计的类分布，而不是均匀分布（Art 最不均衡，sqrt 在 Art 上 +1.2）
        if args.clsw_mode != "inv":
            with torch.no_grad():
                mass = gamma1.sum(dim=1)                                                         # [C] 本批计划在各类上的质量
                w = {"const": torch.ones_like(class_weights), "sqrt": class_weights.sqrt(),
                     "prior": clsw_prior * class_weights}[args.clsw_mode]
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

        model_optimizer.zero_grad()
        loss1.backward()
        model_optimizer.step()

        #########################################################

        clipFeatWGrad = CLIP_model.encode_image(clip_img)
        clipFeatWGrad = clipFeatWGrad / clipFeatWGrad.norm(dim=-1, keepdim=True)

        clipTextFeatGrad = encode_cls_text()
        clipTextFeatGrad = clipTextFeatGrad / clipTextFeatGrad.norm(dim=-1, keepdim=True)
        if args.t_proto:
            clipTextFeatGrad = F.normalize(t_proto, dim=1)

        logits_clip = (100.0 * clipFeatWGrad @ clipTextFeatGrad.T)

        # --knn_mgva：老师（CLIP adapter）学的目标换成学生特征空间近邻平滑后的学生概率
        mgvaProb = knn_smooth(modelFeat, index_img, feat_bank, MB.model_bank, modelProb, args.knn_k) if args.knn_mgva else modelProb
        if args.t_ema:
            t_ema_bank[index_img] = args.t_ema_m * t_ema_bank[index_img] + (1 - args.t_ema_m) * modelProb
            mgvaProb = t_ema_bank[index_img]
        # --consensus teacher / both：老师学的不再是学生的 top-3，而是老师与学生（更新前）按熵加权的专家乘积
        #   第 1 步：学生终点 ≈ 训练前并集 + 0.5，「谁教谁」才是问题；COSMO 消融里共享共识比互相监督 +1.0
        if args.consensus != "none":
            mgvaProb = entropy_poe(clipProb, modelProb)
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
                dyn_acc["keep"] = dyn_acc.get("keep", 0) + int(mgva_mask.sum())             # MGVA 保留的样本数
                dyn_acc["keep_ok"] = dyn_acc.get("keep_ok", 0) + int(((modelProb.argmax(dim=1) == y_b).float() * mgva_mask).sum())

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
                cur = eval_dynamics(feature_net, classifier_net, CLIP_model, cls_text, target_dataloader["test"], args.device, text_fn)
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
    parser.add_argument("--clsw_mode", default="inv", type=str, choices=["inv", "const", "sqrt", "prior"],
                        help="对齐计划的类频率重加权：inv = 1/频率（原方法）；const = 不均衡；sqrt = 1/√频率；prior = CLIP 零样本平均概率 / 频率；"
                             "后三者每步把损失量级调成与 inv 相同")
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
    parser.add_argument("--consensus", default="none", choices=["none", "teacher", "both"],
                        help="none = 原方法（老师学学生 top-3，学生学老师）；teacher = 老师学「老师 × 学生」熵加权专家乘积；both = 学生对齐目标也换成它")
    parser.add_argument("--s_ce", default=0.0, type=float, help="学生对共识 argmax 的硬交叉熵权重；0 = 不用")

    args = parser.parse_args()
    assert not args.knn_mgva or args.knn_k > 0, "--knn_mgva 需要 --knn_k > 0"
    assert not args.log_samples or args.log_dynamics, "--log_samples 需要 --log_dynamics"
    assert args.knn_self_after >= 1.0 or args.knn_k > 0, "--knn_self_after 需要 --knn_k > 0"
    assert not args.src_expert or args.knn_self_after >= 1.0, "--src_expert 不和 --knn_self_after 同时使用"
    assert not (args.t_ema and args.knn_mgva), "--t_ema 不和 --knn_mgva 同时使用"
    assert not (args.t_prompt and args.t_proto), "--t_prompt 不和 --t_proto 同时使用"
    assert not args.no_adapter or args.t_prompt, "--no_adapter 需要 --t_prompt"
    assert not (args.consensus == "both" and args.src_expert), "--consensus both 不和 --src_expert 同时使用"
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
