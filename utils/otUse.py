import ot
import torch

def getGamma(args, clipProb, clipStatistic, reg = 0.2, modelProb = None):
    # 论文式 3：Cv = α·(1 − q^v) + L(q^m, y_j)
    #   第一项用 VLM 概率 q^v（clipProb），第二项是模型预测 q^m 对类 j 的交叉熵 −log q^m_j
    #   原代码第二项用的是 −log(clipProb)，与论文不符
    assert modelProb is not None, "getGamma 需要传入模型预测 modelProb（[C, B]），见论文式 3"
    log_T_prob = torch.log(modelProb + 1e-6)                # [C, B]  log q^m
    cls_dist = -log_T_prob                                  # [C, B]  −log q^m
    feat_dist = 1 - clipProb            # [C, B]
    alpha = 1.0 / feat_dist.max()
    lambda_t = 1.0
    cost_matrix = alpha * feat_dist + lambda_t * cls_dist  # [C, B]
    C, B = cost_matrix.shape
    if clipStatistic is None:
        mu_p = torch.full((C,), 1.0 / C, device=cost_matrix.device).cpu().numpy()
    else:
        mu_p = clipStatistic.cpu().numpy()
    mu_t = torch.full((B,), 1.0 / B, device=cost_matrix.device).cpu().numpy()

    cost_np = cost_matrix.detach().cpu().numpy()
    gamma_np = ot.sinkhorn(mu_p, mu_t, cost_np, reg=reg)
    gamma = torch.tensor(gamma_np, device=cost_matrix.device, dtype=torch.float32)  # [C, B]
    return gamma

def getGamma_self(args, modelProb, model_feat, modelStatistic, classifier_net,reg = 0.2):
    log_T_prob = torch.log(modelProb + 1e-6)                       # [B, C]
    identity = torch.eye(modelProb.size(1)).to(args.device)        # [C, C]
    cls_dist = -torch.matmul(identity, log_T_prob.T)        # [C, B] 
    prototypes = classifier_net.fc.weight.detach()          # shape: [C, d]
    P_expand = prototypes.unsqueeze(1)              # [C, 1, d]
    T_expand = model_feat.unsqueeze(0)              # [1, B, d]
    feat_dist = torch.nn.CosineSimilarity(dim=-1)(P_expand, T_expand)  # [C, B]
    feat_dist = 1 - feat_dist
    alpha = 1.0 / feat_dist.max()
    lambda_t = 1.0
    cost_matrix = alpha * feat_dist + lambda_t * cls_dist  # [C, B]
    C, B = cost_matrix.shape
    if modelStatistic is None:
        mu_p = torch.full((C,), 1.0 / C, device=cost_matrix.device).cpu().numpy()
    else:
        mu_p = modelStatistic.cpu().numpy()
    mu_t = torch.full((B,), 1.0 / B, device=cost_matrix.device).cpu().numpy()
    cost_np = cost_matrix.detach().cpu().numpy()
    gamma_np = ot.sinkhorn(mu_p, mu_t, cost_np, reg=reg)
    gamma = torch.tensor(gamma_np, device=cost_matrix.device, dtype=torch.float32)  # [C, B]
    return gamma, cost_matrix


