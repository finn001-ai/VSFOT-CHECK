import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.utils.weight_norm as weightNorm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

res_dict = {
        "resnet18":models.resnet18, 
        "resnet34":models.resnet34, 
        "resnet50":models.resnet50,
        "resnet101":models.resnet101, 
    }
weight_dict = {
        "resnet18":models.ResNet18_Weights.DEFAULT, 
        "resnet34":models.ResNet34_Weights.DEFAULT, 
        "resnet50":models.ResNet50_Weights.DEFAULT,
        "resnet101":models.ResNet101_Weights.DEFAULT, 
    }


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        return x


class resnet(nn.Module):
    def __init__(self, res_name='resnet50', bottleneck_dim=256):
        super(resnet, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        model_resnet = res_dict[res_name](weights=weight_dict[res_name])
        self.in_features_dim = model_resnet.fc.in_features
        self.backbone = nn.Sequential(*list(model_resnet.children())[:-1])
        self.feat_bottleneck = feat_bottleneck(self.in_features_dim, self.bottleneck_dim )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        feature = self.feat_bottleneck(x)
        return feature




class classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super(classifier, self).__init__()
        self.fc = torch.nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

