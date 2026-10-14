import torch
import torch.nn as nn

def cycDistIdx(cycDistInfo):
    weakModelLabel = cycDistInfo["weakModelLabel"]
    strongModelLabel = cycDistInfo["strongModelLabel"]
    weakVilLabel = cycDistInfo["weakVilLabel"]
    strongVilLabel = cycDistInfo["strongVilLabel"]
    modelLike = weakModelLabel == strongModelLabel
    vilLike = weakVilLabel == strongVilLabel
    xorLike = torch.logical_xor(modelLike, vilLike)
    idxIdenModel = torch.where(torch.logical_and(xorLike, modelLike))
    idxIdenVil = torch.where(torch.logical_and(xorLike, vilLike))
    idxIdentify  = torch.where(~xorLike)
    return idxIdenModel, idxIdenVil, idxIdentify



def cycDistIdx_523(cycDistInfo):
    weakModelLabel = cycDistInfo["weakModelLabel"]
    strongModelLabel = cycDistInfo["strongModelLabel"]
    weakVilLabel = cycDistInfo["weakVilLabel"]
    strongVilLabel = cycDistInfo["strongVilLabel"]
    modelLike = weakModelLabel == strongModelLabel
    vilLike = weakVilLabel == strongVilLabel
    xorLike = torch.logical_xor(modelLike, vilLike)                         # 模型认同vil不认同 或者 vil认同模型不认同

    return {
        "modelLike": torch.where(modelLike == True),                        # 模型一致性认同
        "notmodelLike": torch.where(modelLike == False),                    # 模型不一致
        "vilLike": torch.where(vilLike == True),                            # vil一致性认同
        "notVilLike": torch.where(vilLike == False),                        # vil不一致
        "modelIdent": torch.where(torch.logical_and(xorLike, modelLike)),       # 模型认同, vil不认同
        "vilIdent": torch.where(torch.logical_and(xorLike, vilLike)),           # vil认同, 模型不认同
        "same": torch.where(~xorLike),                                     # 同时认同 或者 同时不认同
        "total": torch.arange(modelLike.shape[0]),
        "empty": torch.tensor([],dtype = int),
    }
