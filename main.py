import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import sys
import clip,copy
import torch
import argparse
from utils.Loss import *
from utils.Project_Record import Project
from utils.Dataset import getDataloaderSFDA5X
from utils.Other import backup_files_and_folders, seed_everything
from utils.Evaluate import modeleval, modeleval_str, lineClassAcc
from model.network import resnet, classifier
from utils.otUse import getGamma, getGamma_self
from utils.MemoryBank import MemoryBank
from utils.adapterVIT import clipAdapter
from model.simsiam import SimSiam, fastSimsiam
import datetime
import json


IMAGEFILE = "/root/user/dataset"

def start(args, target_dataloader, CLIP_model_raw):
    network = torch.load(os.path.join("process", args.sourceStorage, f"Source_{args.source.lower()}", "Source.pt"), weights_only=False)
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
                   {'params': classifier_net.parameters(), 'lr':args.lr * 0.1}]
    model_optimizer = torch.optim.SGD(param_group)
    model_LR = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, max_iter, eta_min=0, last_epoch=- 1, verbose=False)

    CLIP_model = clipAdapter(CLIP_model_raw, args.device, adapterCount = args.adaptercount)
    clip_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CLIP_model.parameters()), lr=5e-5)

    class_prompt = [ f'a photo of a {item}' for item in args.class_info]
    cls_text = clip.tokenize(class_prompt).to(args.device)

    MB = MemoryBank(target_dataloader["test"].dataset.__len__(), args.class_num, device=args.device)
    with torch.no_grad():
        cls_text_features = CLIP_model_raw.encode_text(cls_text)
        MB.initialize(target_dataloader["test"], CLIP_model_raw, feature_net, classifier_net, cls_text_features, args.device)
    SSNB = SimSiam(feature_net).to(args.device)

    model_cm, model_oa, model_aa, model_class_acc = modeleval(args, feature_net, classifier_net, target_dataloader["test"])
    ModellineAcc = lineClassAcc(args, model_class_acc, 0, header=True)
    Project.log(ModellineAcc, filename=f"ModellineAcc.txt", is_print=False, haveTime=False)
    Project.log(f"\n\n\n ==>iter: {iter_num}  " + modeleval_str(model_oa,model_aa,model_cm, args.class_info), filename=f"Detail.txt", is_print = False)
    best_acc = model_oa

    IML = InformationMaximizationLoss(trade_off=args.imloss)

    while iter_num < max_iter:
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

            gamma1 = getGamma(args, clipProb.T, vilMemoryMean, args.otReg)     # [C, B]

        ################################################
        modelFeatGradW = feature_net(weak_img)
        modelLogitsGradW = classifier_net(modelFeatGradW)
        modelProbGrad = torch.softmax(modelLogitsGradW, dim=1)
        gamma2, cost_matrix = getGamma_self(args, modelProbGrad, modelFeatGradW, modelMemoryMean, classifier_net, reg=args.otReg)

        with torch.no_grad():
            total_cls = MB.model_bank.mean(dim=0)
            cls_freq = torch.clamp(total_cls, min=1e-4)
            class_weights = 1.0 / cls_freq
        gamma1 = gamma1 * class_weights.unsqueeze(1)
        loss_align = torch.sum(gamma1 * cost_matrix)

        loss_ctr = fastSimsiam(strong_img, strong2_img, SSNB)

        loss_iml = IML(modelProbGrad)

        progress = iter_num / max_iter
        weight_align = min(1.0, progress * 2)  # [1 -> 2]

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

        top3_probs, top3_indices = torch.topk(modelProb, k=args.topk, dim=1)
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
        clip_optimizer.zero_grad()
        loss_vil.backward()
        clip_optimizer.step()

        ################################################
        model_LR.step()
        MB.update(index_img, vil_prob=clipProb, model_prob=modelProb)

        ################################################
        log_str = f'Task: SFDA, Test Target, {args.source} -> {args.target}, Iter: {iter_num:4d}/{max_iter:4d}; Align: {loss_align.item():.4f}; CTR: {loss_ctr.item():.4f}; IML: {loss_iml.item():.4f}; VLM: {loss_vil.item():.4f};'
        Project.log(log_str)
        if iter_num != 0 and (iter_num % eval_gap == 0 or iter_num == max_iter - 1):
            model_cm, model_oa, model_aa, model_class_acc = modeleval(args, feature_net, classifier_net, target_dataloader["test"])
            ModellineAcc = lineClassAcc(args, model_class_acc, iter_num, header=False)
            Project.log(ModellineAcc, filename=f"ModellineAcc.txt", is_print=False, haveTime=False)
            Project.log(f"\n\n\n ==>iter: {iter_num}  " + modeleval_str(model_oa,model_aa,model_cm, args.class_info), filename=f"Detail.txt", is_print = False)
            acc_total = model_oa
            if acc_total >= best_acc:
                best_acc = acc_total
            log_str = f'Task: SFDA, Test Target, {args.source} -> {args.target}, Iter: {iter_num:4d}/{max_iter:4d}; Now Acc: {acc_total:.4f}; Best Acc: {best_acc:.4f}'
            Project.log(log_str)
        iter_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
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

    args = parser.parse_args()

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
        args.class_info = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']
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

    filename = os.path.basename(sys.argv[0])[:-3] + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if args.name == None:
        Project(project_name=filename, root_name="process")    
    else:
        Project(project_name=args.name, root_name="process")
    args_str = json.dumps(vars(args), ensure_ascii=False, indent=2)
    Project.log(args_str, filename=f"argsInfo.txt", is_print = False)
    Project.log(sys.argv[0])
    Project.log(f"Task: SFDA, Source:{args.source}, Target:{args.target}")
    backup_files_and_folders(Project.root_path, sys.argv[0])

    CLIP_model, CLIP_preprocess = clip.load(args.clipVersion, device=args.device)
    target_dataloader = getDataloaderSFDA5X(args, args.target, CLIP_preprocess, True)
    start(args, target_dataloader, CLIP_model)
