import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
def entropy(predictions,dim=1):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=dim)
    return H

class MemoryBank:
    def __init__(self, dataset_size, prob_dim, device='cpu'):
        self.dataset_size = dataset_size
        self.device = device
        self.class_num = prob_dim
        self.vil_bank = torch.zeros((dataset_size, prob_dim), dtype=torch.float32, device=device)
        self.model_bank = torch.zeros((dataset_size, prob_dim), dtype=torch.float32, device=device)
    def update(self, idx, vil_prob = None, model_prob = None):
        if vil_prob is not None:
            self.vil_bank[idx] = vil_prob.detach().clone().to(self.device).float()
        if model_prob is not None:
            self.model_bank[idx] = model_prob.detach().clone().to(self.device)

    def get(self, idx):
        idx = idx.to(self.device)
        vilTmp = self.vil_bank[idx]
        modelTmp = self.model_bank[idx]
        return vilTmp.mean(dim=0).cuda(), modelTmp.mean(dim=0).cuda()

    def initialize(self, dataloader, vil_model, source_model, classifier, text_features, argsDevice):
        vil_model.eval()
        source_model.eval()
        vilDevice = argsDevice
        sourceDevice = argsDevice
        with torch.no_grad():
            for item in tqdm(dataloader, desc="init mb"):
                target_img = item["img"].to(sourceDevice)
                clip_img = item["clip_img"].to(vilDevice)
                index = item["index"].to(self.device)
                Model_feat = source_model(target_img)
                Model_logits = classifier(Model_feat)
                Model_prob = torch.softmax(Model_logits, dim=1)
                self.model_bank[index] = Model_prob.clone().to(self.device)
                image_features = vil_model.encode_image(clip_img)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                clipModelLogits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                self.vil_bank[index] = clipModelLogits.clone().to(self.device).float()



