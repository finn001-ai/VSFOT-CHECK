import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy

def init_adapter(adapter):
    nn.init.xavier_uniform_(adapter.down.weight, gain=0.1)
    nn.init.constant_(adapter.down.bias, 0)
    nn.init.xavier_uniform_(adapter.up.weight, gain=0.1)
    nn.init.constant_(adapter.up.bias, 0)

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        init_adapter(self)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class ResidualAttentionBlockWithAdapter(nn.Module):
    def __init__(self, orig_block, bottleneck_dim):
        super().__init__()
        self.block = orig_block
        self.adapter = Adapter(orig_block.attn.out_proj.out_features, bottleneck_dim)
    def forward(self, x: torch.Tensor):
        x = self.block(x)
        x = self.adapter(x)
        return x



def insertLayer(clip_model, adapterCount, bottleneck_dim):
    transformer_blocks = clip_model.visual.transformer.resblocks
    tbSize =len(transformer_blocks)
    counter_ = 0
    for blockIdx in range(tbSize, 0, -1):
        transformer_blocks[blockIdx-1] = ResidualAttentionBlockWithAdapter(transformer_blocks[blockIdx-1], bottleneck_dim)
        counter_ += 1
        if counter_ >= adapterCount:
            break
    return clip_model



def clipAdapter(rawClip, device, adapterCount = 1):
    rawClip.float()
    adapterClip = copy.deepcopy(rawClip)
    adapterClip = insertLayer(adapterClip, adapterCount, bottleneck_dim=256)
    adapterClip.to(device).train().float()
    for name, param in adapterClip.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else: 
            param.requires_grad = True
    
    return adapterClip

    
    


    