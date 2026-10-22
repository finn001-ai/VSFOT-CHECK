import os
import random
from PIL import Image, ImageFilter
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def make_dataset(image_list):
    if isinstance(image_list[0], list):
        return image_list
    images = [val.split(",") for val in image_list]
    return images

def RGB_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        


class dataset_norm(Dataset):
    def __init__(self, image_list, isTrian = True):
        self.imgs = make_dataset(image_list)        # (path, label, domain)
        self.imgs_raw = []
        self.imgs_test = []
        self.isTrian = isTrian
        self.get_transform()

    def __getitem__(self, index):
        imgs_path, imgs_label = self.imgs[index]
        img = RGB_loader(imgs_path)

        if self.isTrian:
            return {
                "img":self.train_transform(img),
                "label":int(imgs_label),
            }
        else:
            return {
                "img":self.test_transform(img),
                "label":int(imgs_label),
            }

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )


def get_dataloader_SFDA(args, domain, divide = False, getTrain = True):
    datasetListPath = os.path.join("dataset", args.dataset, f"{domain}.txt")
    with open(datasetListPath, 'r') as file:
        datasetList = file.readlines()
        datasetList = [f"{os.path.join(args.image_root, domain, list_item.split()[0])},{list_item.split()[1]}" for list_item in datasetList]
    random.shuffle(datasetList)
    dataset_length = len(datasetList)
    dataloader_dirt = dict()
    if divide:
        train_num = int(dataset_length * 0.9)
        train_dataset_list = datasetList[:train_num]
        test_dataset_list = datasetList[train_num:]
    else:
        train_dataset_list = datasetList
        test_dataset_list = datasetList  
    if getTrain:
        train_dataset = dataset_norm(image_list = train_dataset_list, isTrian = True)
        dataloader_dirt["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=getattr(args, "workers", None) or 5, pin_memory=True, persistent_workers=True)
    test_dataset = dataset_norm(image_list = test_dataset_list, isTrian = False)
    dataloader_dirt["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=getattr(args, "workers", None) or 5, pin_memory=True, persistent_workers = True)
    full_dataset = dataset_norm(image_list = datasetList, isTrian = False)
    dataloader_dirt["fulltest"] = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=getattr(args, "workers", None) or 5, pin_memory=True, persistent_workers = True)
    return dataloader_dirt


# office-31, office-home
def getDataloaderSource(args, domain):
    datasetListPath = os.path.join("dataset", args.dataset, f"{domain}.txt")
    with open(datasetListPath, 'r') as file:
        datasetList = file.readlines()
        datasetList = [f"{os.path.join(args.image_root, domain, list_item.split()[0])},{list_item.split()[1]}" for list_item in datasetList]
    random.shuffle(datasetList)
    dataset_length = len(datasetList)
    dataloader_dirt = dict()
    train_num = int(dataset_length * 0.9)
    train_dataset_list = datasetList[:train_num]
    test_dataset_list = datasetList[train_num:]
    train_dataset = dataset_norm(image_list = train_dataset_list, isTrian = True)
    dataloader_dirt["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=getattr(args, "workers", None) or 5, pin_memory=True, persistent_workers=True)
    test_dataset = dataset_norm(image_list = test_dataset_list, isTrian = False)
    dataloader_dirt["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=getattr(args, "workers", None) or 5, pin_memory=True, persistent_workers = True)
    full_dataset = dataset_norm(image_list = datasetList, isTrian = False)
    dataloader_dirt["fulltest"] = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=getattr(args, "workers", None) or 5, pin_memory=True, persistent_workers = True)
    return dataloader_dirt




class dataset5X(Dataset):
    def __init__(self, image_list, clipProcess, classDetail, isTraining = True):
        self.imgs = make_dataset(image_list)
        self.clipProcessVal = clipProcess
        self.classDetail = classDetail
        self.isTraining = isTraining
        self.get_transform()
    def __getitem__(self, index):
        imgs_path, imgs_label = self.imgs[index]
        img = RGB_loader(imgs_path)
        returnDict = {
            "label":int(imgs_label),
            "path": imgs_path,
            "index": int(index),
            "label_txt": self.classDetail[int(imgs_label)],
            "clip_img": self.clipProcessVal(img),
        }
        if self.isTraining:
            returnDict["img"] = self.train_transform(img)
            returnDict["strong_img"] = self.mocov2(img)
            returnDict["strong2_img"] = self.mocov2(img)
        else:
            returnDict["img"] = self.test_transform(img)
            returnDict["strong_img"] = self.mocov2(img)
            returnDict["strong2_img"] = self.mocov2(img)
        return returnDict
    def __len__(self):
        return len(self.imgs)


    def get_transform(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
            )
        self.mocov2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def getDataloaderSFDA5X(args, domain, clipProcess = None, getTrain = True):
    datasetListPath = os.path.join("dataset", args.dataset, f"{domain}.txt")
    with open(datasetListPath, 'r') as file:
        datasetList = file.readlines()
        datasetList = [f"{os.path.join(args.image_root, domain, list_item.split()[0])},{list_item.split()[1]}" for list_item in datasetList]
    random.shuffle(datasetList)
    dataloader_dirt = dict()
    if getTrain:
        train_dataset = dataset5X(image_list = datasetList, clipProcess = clipProcess, classDetail = args.class_info, isTraining=True)
        dataloader_dirt["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=getattr(args, "workers", None) or 4, pin_memory=True, persistent_workers=True)
    test_dataset = dataset5X(image_list = datasetList, clipProcess = clipProcess, classDetail = args.class_info, isTraining=False)
    dataloader_dirt["test"] = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, drop_last=False, num_workers=getattr(args, "workers", None) or 4, pin_memory=True, persistent_workers = True)
    return dataloader_dirt