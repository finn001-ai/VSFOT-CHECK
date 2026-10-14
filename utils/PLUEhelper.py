import torch.nn.functional as F
import torch


def refine_predictions(features, banks):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(features, feature_bank, probs_bank)
    return pred_labels, probs, pred_labels_all, pred_labels_hard

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    pred_probs_all = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : 10]
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)
    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)
    _, pred_labels_all = pred_probs_all.max(dim=2)
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)
    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")
    return distances


@torch.no_grad()
def update_labels(banks, idxs, features, probs):
    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-6), dim=axis)

@torch.no_grad()
def createBank(model, classnetwork, banks, test_loader):
    model.eval()
    classnetwork.eval()
    logits, indices, features, gt_labels = [], [], [], []
    for item in test_loader:
        weak_img = item["img"].cuda()
        index_img = item["index"].cuda()
        label_img = item["label"].cuda()
        feats = model(weak_img)
        logic = classnetwork(feats)
        features.append(feats)
        gt_labels.append(label_img)
        logits.append(logic)
        indices.append(index_img)      
    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)
    probs = F.softmax(logits, dim=1)

    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs],
        "probs": probs[rand_idxs],
        "ptr": 0,
    }
    pred_labels, _, _, _ = refine_predictions(features, probs, banks) 
    return _, banks, gt_labels, pred_labels


import torch
from torch import nn
import torch.nn.functional as F
import copy

class AdaMoCo(nn.Module):
    def __init__(self, src_model, features_length, num_classes, dataset_length, temporal_length, device = "cuda"):
        super(AdaMoCo, self).__init__()
        self.m = 0.999
        self.first_update = True
        self.src_model = src_model
        self.momentum_model = copy.deepcopy(src_model)
        self.momentum_model.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0

        self.T_moco = 0.07
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer("labels", torch.randint(0, num_classes, (self.K,)))
        self.register_buffer("idxs", torch.randint(0, dataset_length, (self.K,)))
        self.register_buffer("mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length)))
        self.features = F.normalize(self.features, dim=0)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.mem_labels = self.mem_labels.to(device)
        self.idxs = self.idxs.to(device)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.src_model.parameters(), self.momentum_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels):
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.queue_ptr = end % self.K
        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        return self.features, self.labels


    def forward(self, feats_q, im_k):
        q = F.normalize(feats_q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.momentum_model(im_k)
            k = F.normalize(k, dim=1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])
        logits_ins = torch.cat([l_pos, l_neg], dim=1)
        logits_ins /= self.T_moco
        return None, None, logits_ins, k