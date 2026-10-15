import torch
import numpy
import random
import math
import shutil
import os
import sys
def seed_everything(seed = 0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def format_dict(data):
    formatted_string = ", ".join(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}" for key, value in data.items())
    formatted_string += f", total: {sum(data.values()):.3f}"
    return formatted_string

def format_acc(data, args):
    log = f"src_acc: {data['src_acc'][0]:.3f} -> "
    acc_total, acc_every = data['tar_acc']
    every_foramt = ", ".join( [f"{key[0].upper()}: {value:.3f}" for key, value in acc_every.items()])
    log += f"{ every_foramt }, "
    log += f"total: {acc_total:.3f}"
    return log

def backup_files_and_folders(backup_root, main_file, target_folders = ["model", "utils"]):
    backup_dir = os.path.join(backup_root, f'project_backup')
    os.makedirs(backup_dir, exist_ok=True)
    for item in os.listdir('.'):
        if os.path.isfile(item) and (item in main_file):
            shutil.copy2(item, backup_dir)
        elif os.path.isdir(item) and (item in target_folders):
            shutil.copytree(item, os.path.join(backup_dir, item))

def update_ema(ema_model, activate_model, alpha=0.7):
    for ema_param, param in zip(ema_model.parameters(), activate_model.parameters()):
        ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data

def load_class_names(path, class_num=None):
    """从 classname.txt 读取类名，每行一个，行号（从 0 开始）即标签编号"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到类名文件 {path}，请先把 classname.txt 链接到该位置（见 link_data.sh）")
    with open(path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if class_num is not None and len(names) != class_num:
        raise ValueError(f"{path} 中有 {len(names)} 个类名，应为 {class_num} 个")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} 中有重复的类名")
    return names


def check_class_names(class_names, list_paths):
    """
    核对类名顺序和图片列表的标签是否一致：
    每行「路径 标签」中，路径倒数第二段（图片所在的类文件夹名）必须等于 class_names[标签]。
    不一致时 CLIP 提示词会配错类别，训练不会报错但结果会被拉低，所以这里直接报错。
    """
    for path in list_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rel, label = line.split()[0], int(line.split()[1])
                folder = rel.replace("\\", "/").split("/")[-2]
                if label >= len(class_names) or class_names[label] != folder:
                    want = class_names[label] if label < len(class_names) else None
                    raise ValueError(f"{path}：标签 {label} 的图片在文件夹 {folder}，而类名文件中第 {label} 行为 {want}")
