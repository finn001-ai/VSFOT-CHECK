import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import sys
import copy
import torch
import argparse
from utils.Loss import *
from utils.Project_Record import Project
from utils.Dataset import get_dataloader_SFDA
from utils.Other import backup_files_and_folders, seed_everything
from utils.Evaluate import test_domain_sfda
from model.network import resnet, classifier
import datetime
import json


IMAGEFILE = "dataset"    # 相对项目根目录；图片通过软链接放在 dataset/<数据集>/<域>/ 下

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer



def start(args, source_dataloader):
    feature_net = resnet(res_name=args.arch).cuda()
    classifier_net = classifier(class_num = args.class_num).cuda()

    param_group = [
        {'params': feature_net.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': feature_net.feat_bottleneck.parameters(), 'lr': args.lr},
        {'params': classifier_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    best_acc = 0
    iter_num = 0
    max_iter = args.epochs * len(source_dataloader["train"])
    interval_iter = max_iter // 10

    while iter_num < max_iter:
        feature_net.train()
        classifier_net.train()
        try:
            item = next(iter_source)
            inputs_source, labels_source = item["img"].cuda(), item["label"].cuda()
        except:
            iter_source = iter(source_dataloader["train"])
            item = next(iter_source)
            inputs_source, labels_source = item["img"].cuda(), item["label"].cuda()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # network
        outputs_source = classifier_net(feature_net(inputs_source))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source, labels_source) 
        log_str = 'Task: Train Source-{}, Iter:{:4d}/{:4d}; Loss: {:.4f}'.format(args.source, iter_num, max_iter, classifier_loss.item())
        Project.log(log_str)           
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()


        # eval & save
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            acc_total = test_domain_sfda(feature_net, classifier_net, source_dataloader["test"])
            if acc_total >= best_acc:
                best_acc = acc_total
                best_feature_net = copy.deepcopy(feature_net.state_dict())
                best_classifier_net = copy.deepcopy(classifier_net.state_dict())
            log_str = 'Task: Train Source-{}, Iter:{:4d}/{:4d}; Now Acc: {:.4f}; Best Acc: {:.4f}'.format(args.source, iter_num, max_iter, acc_total, best_acc)
            Project.log(log_str)

    Network = {
        "best_feature": best_feature_net,
        "best_classifier": best_classifier_net,
        "best_acc": best_acc
    }
    torch.save(Network, os.path.join(Project.root_path, f"Source.pt"))
    return Network



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=None, help="DataLoader 进程数；默认沿用原设置（4 / 5）。VisDA / DomainNet 图片大，数据加载是瓶颈时调大")
    parser.add_argument('--source', type=str, default="clipart")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--dataset', type=str, default="DomainNet")
    parser.add_argument("--lr",default=1e-2,type=float,metavar="LR",help="initial learning rate",dest="lr")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.dataset == "OFFICE31":
        args.class_num = 31
        args.domains = ["amazon", "dslr", "webcam"]
        args.class_info = ['back_pack','bike','bike_helmet','bookcase','bottle','calculator','desktop_computer','desk_chair','desk_lamp','file_cabinet','headphones','keyboard','laptop_computer','letter_tray','mobile_phone','monitor','mouse','mug','paper_notebook','pen','phone','printer','projector','punchers','ring_binder','ruler','scissors','speaker','stapler','tape_dispenser','trash_can']
    elif args.dataset == "OfficeHomeDataset":
        args.class_num = 65
        args.source = {"art":"Art","clipart":"Clipart","product":"Product","realworld":"Real World"}[args.source]
        # args.target = {"art":"Art","clipart":"Clipart","product":"Product","realworld":"Real World"}[args.target]
        args.domains = ["Art", "Clipart", "Product", "Real World"]
        args.class_info = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']
    elif args.dataset == "VisDA":
        args.class_num = 12
        args.domains = ["synthesis", "real"]
        args.class_info = ['aeroplane','bicycle','bus','car','horse','knife','motorcycle','person','plant','skateboard','train','truck']
    elif args.dataset == "DomainNet":
        args.class_num = 126
        args.domains = ["clipart", "painting", "real", "sketch"]
        args.class_info = ['aircraft_carrier', 'alarm_clock', 'ant', 'anvil', 'asparagus', 'axe', 'banana', 'basket', 'bathtub', 'bear', 'bee', 'bird', 'blackberry', 'blueberry', 'bottlecap', 'broccoli', 'bus', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon', 'canoe', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'coffee_cup', 'compass', 'computer', 'cow', 'crab', 'crocodile', 'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck', 'dumbbell', 'elephant', 'eyeglasses', 'feather', 'fence', 'fish', 'flamingo', 'flower', 'foot', 'fork', 'frog', 'giraffe', 'goatee', 'grapes', 'guitar', 'hammer', 'helicopter', 'helmet', 'horse', 'kangaroo', 'lantern', 'laptop', 'leaf', 'lion', 'lipstick', 'lobster', 'microphone', 'monkey', 'mosquito', 'mouse', 'mug', 'mushroom', 'onion', 'panda', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'pig', 'pillow', 'pineapple', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone', 'screwdriver', 'sea_turtle', 'see_saw', 'sheep', 'shoe', 'skateboard', 'snake', 'speedboat', 'spider', 'squirrel', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'swan', 'table', 'teapot', 'teddy-bear', 'television', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'tiger', 'toe', 'train', 'truck', 'umbrella', 'vase', 'watermelon', 'whale', 'zebra']
    
    assert args.source in args.domains, f"error dataset or domai: {args.source} not in {args.dataset}"
    args.image_root = os.path.join(IMAGEFILE, args.dataset)
    args.device = f"cuda:0"

    filename = os.path.basename(sys.argv[0])[:-3] + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if args.name == None:
        Project(project_name=filename, root_name="runs")    
    else:
        Project(project_name=args.name, root_name="runs")
    args_str = json.dumps(vars(args), ensure_ascii=False, indent=2)
    Project.log(args_str, filename=f"argsInfo.txt", is_print = False)
    Project.log(sys.argv[0])
    Project.log(f"Train Source:{args.source}")
    # backup_files_and_folders(Project.root_path, sys.argv[0])
    source_dataloader = get_dataloader_SFDA(args, domain = args.source, divide = True, getTrain = True)

    Project.log("Start")
    network = start(args, source_dataloader)

    trainDomain = args.source
    otherDomain = args.domains.copy()
    otherDomain.remove(trainDomain)
    for nowDomain in otherDomain:
        feature_net = resnet(res_name=args.arch).cuda()
        feature_net.load_state_dict(network["best_feature"])
        classifier_net = classifier(class_num = args.class_num).cuda()
        classifier_net.load_state_dict(network["best_classifier"])
        testDomain_dataloader = get_dataloader_SFDA(args, domain = nowDomain, divide = True, getTrain = False)
        acc_total = test_domain_sfda(feature_net, classifier_net, testDomain_dataloader['fulltest'])
        log_str = f'Task: Test Source: {args.source}, Target: {nowDomain}, Acc: {acc_total:.3f}'
        Project.log(log_str)