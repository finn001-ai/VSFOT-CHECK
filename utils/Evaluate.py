import torch
import numpy as np
import clip
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score



@torch.no_grad()
def test_domain_sfda(feature, classification, dataloader):
    feature.eval()
    classification.eval()
    acc_total = []
    for data in dataloader:
        test_img = data["img"].cuda()
        test_label = data["label"].cuda()
        test_output = classification(feature(test_img))
        test_predict = torch.argmax(torch.softmax(test_output,dim=1),dim=1)
        tmp = (test_label == test_predict).float().cpu().tolist()
        acc_total.extend(tmp)
    acc_total = np.mean(acc_total)
    return acc_total

def lineClassAcc(args, classac, iter_num, header=False):
    col_width = 12
    fmt = f"{{:>{col_width}.4f}}"
    if header:
        header = ["Iter".ljust(col_width)] + [cls.ljust(col_width) for cls in args.class_info]
        out = "".join(header) + "\n"
    else:
        out = ""
    epoch_str = str(iter_num).ljust(col_width)
    acc_str_list = [fmt.format(a) for a in classac]
    line = [epoch_str] + acc_str_list
    out += "".join(line)
    return out

@torch.no_grad()
def modeleval(args, feature, classification, dataloader):
    feature.eval()
    classification.eval()
    all_groundtruth = []
    all_modelpreds = []

    for data in dataloader:
        img = data["img"].to(args.device)
        test_label = data["label"].to(args.device)
        test_output = classification(feature(img))
        model_predict = torch.argmax(torch.softmax(test_output, dim=1), dim=1)
        all_groundtruth.extend(test_label.cpu().numpy())
        all_modelpreds.extend(model_predict.cpu().numpy())

    model_cm = confusion_matrix(all_groundtruth, all_modelpreds)
    model_oa = accuracy_score(all_groundtruth, all_modelpreds)
    model_aa = np.mean(model_cm.diagonal() / model_cm.sum(axis=1))
    model_class_acc = model_cm.diagonal() / model_cm.sum(axis=1)

    return model_cm, model_oa, model_aa, model_class_acc

def modeleval_str(ModelOA,ModelAA,ModelCM,className):
    output =  "==================> Model <==================\n"
    output +=  f"Overall Accuracy (OA): {ModelOA:.4f}\n"
    output +=  f"Average Accuracy (AA): {ModelAA:.4f}\n"
    output +=  f"Confusion Matrix (cm):\n\t"
    max_width = max(max(len(str(cell)) for cell in row) for row in ModelCM)
    max_width = max(max_width, max(len(name) for name in className))
    cell_format = "{:>" + str(max_width) + "}"
    output += " " * max_width + " | " + " | ".join(cell_format.format(name) for name in className) + "\n"
    output += "-" * (max_width + 1) + "+" + "-" * (len(className) * (max_width + 3) - 1) + "\n"
    for i, row in enumerate(ModelCM):
        output += cell_format.format(className[i]) + " | " + " | ".join(cell_format.format(cell) for cell in row) + "\n"
    return output

@torch.no_grad()
def testCLIP(model, dataloader, args):
    model.eval()
    text = [ f'a photo of a {item}' for item in args.class_info]
    text = clip.tokenize(text).to(args.device)
    all_preds = []
    all_labels = []
    for data in tqdm(dataloader):
        images = data["clip_img"].to(args.device)
        labels = data["label"].to(args.device)
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        preds = similarity.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    cm = confusion_matrix(all_labels, all_preds)
    oa = np.mean(all_preds == all_labels)
    class_accuracies = []
    for i in range(args.class_num):
        idx = all_labels == i
        if np.sum(idx) > 0:
            class_acc = np.mean(all_preds[idx] == all_labels[idx])
            class_accuracies.append(class_acc)
    aa = np.mean(class_accuracies)
    return oa, aa, cm

def info2str(OA,AA,CM, className = None):
    output =  f"Overall Accuracy (OA): {OA:.4f}\n"
    output +=  f"Average Accuracy (AA): {AA:.4f}\n"
    output +=  f"Confusion Matrix (cm):\n\t"
    if className == None:
        for row in CM:
            output += " ".join(map(str, row)) + "\n\t"
        return output
    else:
        max_width = max(max(len(str(cell)) for cell in row) for row in CM)
        max_width = max(max_width, max(len(name) for name in className))
        cell_format = "{:>" + str(max_width) + "}"
        output += " " * max_width + " | " + " | ".join(cell_format.format(name) for name in className) + "\n"
        output += "-" * (max_width + 1) + "+" + "-" * (len(className) * (max_width + 3) - 1) + "\n"
        for i, row in enumerate(CM):
            output += cell_format.format(className[i]) + " | " + " | ".join(cell_format.format(cell) for cell in row) + "\n"
        return output

