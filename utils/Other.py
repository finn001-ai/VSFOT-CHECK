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