"""
CLIP 零样本准确率测试（不训练）

目的：核对我们的 CLIP 起点是否和论文一致。
论文表 4：CLIP（ViT-B/32）在 Office-Home 四个域上的零样本平均准确率为 79.04%。

测试两种提示词写法：
  underscore  原代码写法：a photo of a Alarm_Clock
  space       下划线换成空格：a photo of a Alarm Clock
（CLIP 分词器会自动转小写，所以大小写不影响结果）

用法（在项目根目录下，已执行过 scripts/link_data.sh）：
  python tools/zeroshot_clip.py
输出：屏幕打印结果表，并保存到 runs/zeroshot_results/<日期>/result.md、result.json
"""
import os as _os, sys as _sys
# 本脚本在 tools/ 下：把项目根目录加入 import 路径（utils / model），并切换到根目录（代码用相对路径读 dataset/、runs/）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import argparse
import datetime

import torch
import clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.Dataset import RGB_loader
from utils.Other import load_class_names, check_class_names

DOMAINS = ["Art", "Clipart", "Product", "Real World"]
PAPER_AVG = 79.04          # 论文表 4，CLIP ViT-B/32 在 Office-Home 的零样本平均准确率
TEMPLATE = "a photo of a {}"


class ListDataset(Dataset):
    """按 dataset/<数据集>/<域>.txt 读取图片，路径和训练代码的拼法一致"""
    def __init__(self, root, domain, preprocess):
        self.items = []
        with open(os.path.join("dataset", "OfficeHomeDataset", f"{domain}.txt"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rel, label = line.split()[0], int(line.split()[1])
                    self.items.append((os.path.join(root, domain, rel), label))
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.preprocess(RGB_loader(path)), label




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clip", type=str, default="ViT-B/32")
    parser.add_argument("--root", type=str, default="dataset/OfficeHomeDataset", help="图片根目录（软链接所在）")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    a = parser.parse_args()
    device = f"cuda:{a.device}"

    model, preprocess = clip.load(a.clip, device=device)
    model.float().eval()     # 和训练代码一致：clipAdapter 会把 CLIP 转成 float32

    # 和 main.py 相同：类名从 <image_root>/classname.txt 读取，并核对与四个域图片列表的标签一致（不一致直接报错）
    class_names = load_class_names(os.path.join(a.root, "classname.txt"), 65)
    check_class_names(class_names, [os.path.join("dataset", "OfficeHomeDataset", f"{d}.txt") for d in DOMAINS])
    print(f"==> 类名来自 {a.root}/classname.txt，与四个域的标签核对一致；第 58-63 类: {class_names[58:64]}")
    variants = {
        "underscore": [TEMPLATE.format(c) for c in class_names],
        "space": [TEMPLATE.format(c.replace("_", " ")) for c in class_names],
    }
    for k, v in variants.items():
        # 提示词必须各不相同且包含类名，否则 CLIP 无法区分类别
        assert len(set(v)) == len(v) and all(p != TEMPLATE.format("") for p in v), f"[{k}] 提示词有空的或重复的"
        print(f"[{k}] 示例提示词: {v[:3]} ... {v[-2:]}")

    with torch.no_grad():
        text_feats = {}
        for k, prompts in variants.items():
            t = model.encode_text(clip.tokenize(prompts).to(device))
            text_feats[k] = t / t.norm(dim=-1, keepdim=True)

        result = {k: {} for k in variants}
        for d in DOMAINS:
            ds = ListDataset(a.root, d, preprocess)
            loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
            correct = {k: 0 for k in variants}
            for img, label in tqdm(loader, desc=d):
                f = model.encode_image(img.to(device))
                f = f / f.norm(dim=-1, keepdim=True)
                label = label.to(device)
                for k in variants:
                    correct[k] += ((f @ text_feats[k].T).argmax(dim=1) == label).sum().item()
            for k in variants:
                result[k][d] = 100.0 * correct[k] / len(ds)
            print(f"{d:10s} n={len(ds):5d}  " + "  ".join(f"{k}={result[k][d]:.2f}" for k in variants))

    # ---------------- 结果表 ----------------
    lines = [f"CLIP {a.clip} 零样本准确率（%），模板 \"{TEMPLATE}\"", "",
             "| 提示词 | " + " | ".join(DOMAINS) + " | 平均 | 论文表 4 平均 |",
             "|---|" + "---|" * (len(DOMAINS) + 2)]
    for k in variants:
        avg = sum(result[k].values()) / len(DOMAINS)
        result[k]["avg"] = avg
        lines.append(f"| {k} | " + " | ".join(f"{result[k][d]:.2f}" for d in DOMAINS) + f" | {avg:.2f} | {PAPER_AVG:.2f} |")
    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join("runs/zeroshot_results", datetime.datetime.now().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"clip": a.clip, "template": TEMPLATE, "paper_avg": PAPER_AVG, "result": result}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存到 {out}/")


if __name__ == "__main__":
    main()
