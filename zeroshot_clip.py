"""
CLIP 零样本准确率测试（不训练）

目的：核对我们的 CLIP 起点是否和论文一致。
论文表 4：CLIP（ViT-B/32）在 Office-Home 四个域上的零样本平均准确率为 79.04%。

测试两种提示词写法：
  underscore  原代码写法：a photo of a Alarm_Clock
  space       下划线换成空格：a photo of a Alarm Clock
（CLIP 分词器会自动转小写，所以大小写不影响结果）

用法（在项目根目录下，已执行过 link_data.sh）：
  python zeroshot_clip.py
输出：屏幕打印结果表，并保存到 zeroshot_results/<日期>/result.md、result.json
"""
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


# 和 main.py 中 OfficeHomeDataset 的 args.class_info 完全相同（标签编号即列表下标）
# 注意：不能从图片列表的路径里解析类名 —— 云端 *_list.txt 的路径格式和项目自带的 txt 不同
CLASS_NAMES = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']


def check_label_names(domain):
    """
    核对图片列表中「标签编号 -> 类文件夹名」是否和 CLASS_NAMES 一致。
    类名取路径的倒数第二段（图片所在文件夹），不受路径是否带前缀 / 绝对路径影响。
    不一致时，提示词会配错类别，CLIP 准确率被压低但不会报错，所以必须检查。
    """
    seen = {}
    with open(os.path.join("dataset", "OfficeHomeDataset", f"{domain}.txt"), encoding="utf-8") as f:
        first = None
        for line in f:
            if line.strip():
                rel, label = line.split()[0], int(line.split()[1])
                first = first or rel
                parts = rel.replace("\\", "/").split("/")
                seen.setdefault(label, set()).add(parts[-2] if len(parts) >= 2 else "?")
    bad = [(k, sorted(v), CLASS_NAMES[k] if k < len(CLASS_NAMES) else None)
           for k, v in sorted(seen.items()) if v != {CLASS_NAMES[k] if k < len(CLASS_NAMES) else None}]
    print(f"[{domain}] 列表第一行路径: {first}  标签数: {len(seen)}  与 CLASS_NAMES 不一致: {len(bad)}")
    for k, v, want in bad[:10]:
        print(f"    标签 {k}: 列表中的文件夹 {v}，CLASS_NAMES 为 {want}")
    return len(bad)


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

    class_names = CLASS_NAMES
    n_bad = sum(check_label_names(d) for d in DOMAINS)
    print(f"==> 标签与类名核对：共 {n_bad} 处不一致" + ("（提示词会配错类别！）" if n_bad else "（一致）"))
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

    out = os.path.join("zeroshot_results", datetime.datetime.now().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"clip": a.clip, "template": TEMPLATE, "paper_avg": PAPER_AVG, "result": result}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存到 {out}/")


if __name__ == "__main__":
    main()
