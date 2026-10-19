import torch.nn as nn
import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch
import ot  # pip install POT


CELoss = nn.CrossEntropyLoss()

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        
def one_hot_CrossEntropy(x, y):
    P_i = torch.nn.functional.softmax(x, dim=1)
    loss = y * torch.log(P_i + 0.0000001)
    loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
    return loss

def contrastive_loss_default(logits):
    # 普通的对比学习策略, 看看效果
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    return CELoss(logits, labels)

def contrastive_loss_v1MultiLabel(logits, queueLabel, batchLabel, unconfFlag):
    '''
    筛选置信度
        1. 正样本: 自己和自己
        2. 负样本: 剔除高置信样本相同类别标签的样本
    1. 对于当前batch需要筛选高低置信
    2. 对于低置信的只有自己和自己
    3. 对比高置信的剔除自己和queue中类别一样的样本

    queueLabel:
        为queue的label, 对于simulation使用标签, 对于target使用伪标签当不置信时使用class_class即不可能出现的类别
    '''
    conf_flag = torch.ones( [logits.shape[0]] ).cuda()
    unconf_index = torch.nonzero( batchLabel == unconfFlag ).flatten()
    conf_flag[unconf_index] -= 1
    conf_index = torch.nonzero(conf_flag == 1).flatten()

    mask = torch.ones_like(logits, dtype=bool)  # 1为有效, 0为无效
    labelCompare = (batchLabel.unsqueeze(1) != queueLabel.unsqueeze(0))  # 不等于为T即不等于要考虑, 等于的要忽略

    mask[conf_index, 1:] = labelCompare[conf_index,:]
    logits = torch.where(mask, logits, torch.tensor([float("-inf")]).cuda())

    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    return CELoss(logits, labels)




def orthogonality_loss(W, normalize=True):
    """
    计算分类器参数的正交性损失项：||W^T W - I||^2

    参数:
        W: Tensor of shape [num_classes, feature_dim]
           一般是分类器的权重，例如 model.fc.weight
        normalize: 是否将每个类别向量归一化到单位范数（cosine 正交）

    返回:
        loss: scalar tensor, 正交损失
    """
    if normalize:
        W = F.normalize(W, p=2, dim=1)  # 每行单位化，方向更重要
    WT_W = torch.matmul(W, W.t())       # [C, C] Gram matrix
    I = torch.eye(W.size(0)).to(W.device)
    loss = F.mse_loss(WT_W, I)
    return loss



def optimal_transport_loss(T_features, classifier_net, reg=0.05, detach_proto=True):
    """
    使用 Sinkhorn 最优传输计算目标域特征与类别原型之间的对齐损失

    Args:
        T_features (Tensor): [N, D] 目标域特征
        classifier_net (nn.Module): 包含 fc 层的分类器网络
        reg (float): Sinkhorn 正则化系数
        detach_proto (bool): 是否从计算图中分离原型（防止更新分类器参数）

    Returns:
        loss (Tensor): OT对齐损失，可反向传播
    """
    N, D = T_features.shape
    class_prototypes = classifier_net.fc.weight  # [C, D]
    if detach_proto:
        class_prototypes = class_prototypes.detach()
    C = class_prototypes.shape[0]
    cost_matrix = torch.cdist(T_features, class_prototypes, p=2).detach().cpu().numpy()  # [N, C]

    # 均匀分布
    a = ot.unif(N)  # [N]
    b = ot.unif(C)  # [C]

    # Sinkhorn OT: 得到传输矩阵 [N, C]
    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg)  # [N, C], numpy

    # 转为 torch 张量，发现T全0，存在问题
    T = torch.tensor(transport_plan, dtype=T_features.dtype, device=T_features.device)  # [N, C]

    # 加权原型表示 [N, D]
    aligned_features = T @ class_prototypes  # [N, D]

    # L2 loss between target feature and aligned prototype
    loss = F.mse_loss(T_features, aligned_features)

    return loss





def optimal_transport_loss_V2(T_features, classifier_net, reg=0.05, detach_proto=True):
    N, D = T_features.shape
    class_prototypes = classifier_net.fc.weight  # [C, D]
    if detach_proto:
        class_prototypes = class_prototypes.detach()
    C = class_prototypes.shape[0]

    # Step 1: Compute cost matrix (normalize scale)
    cost_matrix = torch.cdist(T_features, class_prototypes, p=2)  # [N, C]
    cost_matrix = cost_matrix / cost_matrix.max()  # Normalize to [0,1]
    cost_matrix_np = cost_matrix.detach().cpu().numpy()

    # Step 2: Uniform marginals
    a = ot.unif(N)
    b = ot.unif(C)

    # Step 3: Solve Kantorovich OT with entropic regularization
    transport_plan = ot.sinkhorn(a, b, cost_matrix_np, reg)  # [N, C]
    T = torch.tensor(transport_plan, dtype=T_features.dtype, device=T_features.device)

    # Step 4: Map features using transport plan
    aligned_features = T @ class_prototypes  # [N, D]

    # Step 5: Compute alignment loss
    loss = F.mse_loss(T_features, aligned_features)
    return loss





import torch
import torch.nn.functional as F

def ot_prototype_regression_loss(image_features: torch.Tensor,
                                  class_prototypes: torch.Tensor,
                                  T: torch.Tensor) -> torch.Tensor:
    """
    原型增强监督：目标样本特征拟合由OT加权的类原型
        1. 对于每个目标领域样本，使用OT分配一个类别原型
        2. 构建加权原型表示
        3. 引导这个原型对齐到加权中心
    
    Args:
        image_features: [B, D] 目标样本特征
        class_prototypes: [C, D] 源类别原型
        T: [C, B] OT传输矩阵（source类 → target样本）
        gamma: 权重因子
        
    Returns:
        loss: scalar 回归损失
    """
    soft_labels = T.transpose(0, 1)  # [B, C]
    weighted_prototypes = soft_labels @ class_prototypes  # 每个目标样本的类原型加权表示
    loss = F.mse_loss(image_features, weighted_prototypes, reduction='mean')
    return loss



def ot_weighted_soft_ce_loss(logits: torch.Tensor,
                              T: torch.Tensor) -> torch.Tensor:
    """
    OT样本重加权交叉熵损失
        1. 不转换为伪标签，直接将传输质量转换为样本重要性
    
    Args:
        logits: [B, C] 目标模型输出
        T: [C, B] OT传输矩阵（source类 → target样本）
        gamma: 权重因子
        
    Returns:
        loss: scalar 加权交叉熵损失
    """
    soft_labels = T.transpose(0, 1)  # [B, C]
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_labels * log_probs).sum(dim=1).mean()

    return loss


import torch
import torch.nn.functional as F

def ot_soft_label_ce_loss(logits: torch.Tensor,
                          T: torch.Tensor,
                          gamma: float = 1.0) -> torch.Tensor:
    """
    使用 OT 得到的 soft label 训练目标模型（交叉熵损失）

    Args:
        logits: [B, C]，目标模型输出（未经过 softmax）
        T: [C, B]，最优传输矩阵（源类别 → 目标样本）
        gamma: 损失加权因子

    Returns:
        loss: scalar，soft label 交叉熵损失
    """
    # Transpose T to get soft labels: [B, C]
    soft_labels = T.transpose(0, 1)  # [B, C]

    # Log-softmax over logits
    log_probs = F.log_softmax(logits, dim=1)  # [B, C]

    # Compute soft cross-entropy: L = -sum(p * log q)
    loss = -(soft_labels * log_probs).sum(dim=1).mean()

    return gamma * loss

def consistency_loss(logits_weak, logits_strong):
    log_probs_weak = F.log_softmax(logits_weak, dim=1)
    probs_strong = F.softmax(logits_strong, dim=1).detach()  # 停止梯度传播，防止反向影响强增强分支

    # KL 散度计算
    loss_consistency = KL_Loss(log_probs_weak, probs_strong)

    return loss_consistency


KL_Loss = nn.KLDivLoss(reduction="batchmean")
Cons_Loss = F.mse_loss



import torch.nn.functional as F

def contD(p, z):
    """Negative Cosine Similarity"""
    z = z.detach()  # Stop-gradient
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 1-(p * z).sum(dim=1).mean()

def simsiam_loss(p1, z2, p2, z1):
    """
    p1: prediction from view 1
    z2: target from view 2
    p2: prediction from view 2
    z1: target from view 1
    """
    loss = contD(p1, z2) / 2 + contD(p2, z1) / 2
    return loss




def CTRR_Loss(predict_1, projection_2, output_logic, tau = 0.8):
    batch_size = predict_1.shape[0]
    predict_1 = torch.clamp(predict_1, 1e-4, 1.0 - 1e-4)           # q_1【batch, Class_num】
    projection_2 = torch.clamp(projection_2, 1e-4, 1.0 - 1e-4)     # z_2【batch, Projection】
    contrast_1 = torch.matmul(predict_1, projection_2.t())         # <q_1 · z_2> 【batch, batch】 相似度【网络输出完成-nn.functional.normalize】
    tmp_A = -contrast_1 * torch.zeros(batch_size, batch_size).fill_diagonal_(1).cuda()             # log(1-<q_1·z_2>
    tmp_B = ((1-contrast_1).log()) * torch.ones(batch_size, batch_size).fill_diagonal_(0).cuda()
    contrast_1 = tmp_A + tmp_B                                                                     # 处理对角线防止log内部为0
    contrast_logits = 2 + contrast_1                               # 【batch, batch】
    # 根据预测结果选择样本
    soft_targets = torch.softmax(output_logic, dim=1)                                   # 得到预测的softmax
    contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()       # 对预测结果进行点乘,得到相似度
    contrast_mask.fill_diagonal_(1)                                                     # 填充对角线
    pos_mask = (contrast_mask >= tau).float()                                           # 获取指示掩码
    contrast_mask = contrast_mask * pos_mask                                           # 获取指示掩码
    contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True)                 # 这里的权重是考虑同类样本数目了, 比如x_1与x'_2~x'_5对比,得到累加和以后处于对比样本个数保持均衡

    loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)                     # 对每个样本计算sum, 对所有样本取平均【contrast_logits遮盖相似度小的样本】
    return loss_ctr



def contrastive_loss(logits_ins, pseudo_labels, mem_labels, device = "cuda"):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).to(device)

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) 
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(device))

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss



import torch
import torch.nn as nn

class InformationMaximizationLoss(nn.Module):
    def __init__(self, trade_off=1.0, ent_weight=1.0):
        """
        信息最大化损失函数：
        - 最小化预测熵（单个样本）
        - 最大化类别熵（整个 batch 的平均预测）

        Args:
            trade_off: 控制 diversity loss 的权重
            ent_weight: 样本熵项的权重（1.0 = 原方法；0 = 消融，只保留 diversity）
        """
        super(InformationMaximizationLoss, self).__init__()
        self.trade_off = trade_off
        self.ent_weight = ent_weight

    def forward(self, prob):
        """
        Args:
            prob (Tensor): shape [B, C]，已经经过 softmax 的预测概率
        Returns:
            loss (Tensor): scalar，信息最大化损失
        """
        # 最小化样本预测熵
        entropy_loss = -torch.mean(torch.sum(prob * torch.log(prob + 1e-5), dim=1))

        # 最大化类别熵（即最小化负熵）
        mean_prob = torch.mean(prob, dim=0)
        diversity_loss = torch.sum(mean_prob * torch.log(mean_prob + 1e-5))

        loss = self.ent_weight * entropy_loss - self.trade_off * diversity_loss
        return loss



def get_entropy_weight(prob):
    entropy = - torch.sum(prob * torch.log(prob + 1e-5), dim=1)
    weight = 1 - entropy / np.log(prob.size(1))  # normalize to [0, 1]
    return weight.detach()



def unit_norm_regularization(classifier, lambda_reg=1e-2):
    weights = classifier.fc.weight
    norms = torch.norm(weights, p=2, dim=1)
    return lambda_reg * torch.mean((norms - 1) ** 2)



def classifier_repulsion_regularization(classifier, lambda_reg=1e-2):
    W = classifier.fc.weight  # [C, d]
    W_norm = F.normalize(W, dim=1)
    sim = W_norm @ W_norm.T  # cosine similarity
    mask = ~torch.eye(W.size(0), dtype=torch.bool, device=W.device)
    off_diag_sim = sim[mask]
    return lambda_reg * torch.mean(torch.exp(off_diag_sim))  # 越相似惩罚越大