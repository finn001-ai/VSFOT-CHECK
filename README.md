## Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport

### Abstract

Unsupervised domain adaptation transfers knowledge from a labeled source domain to an unlabeled target domain. When source data cannot be accessed, source-free domain adaptation (SFDA) becomes a practical alternative. However, existing SFDA methods mainly rely on pseudo-label based self-training, which often accumulates noise and bias under large domain gaps. We propose VSFOT, a framework that leverages a pretrained Vision-Language Model (VLM) to guide optimal transport (OT) alignment between target features and source prototypes. Instead of relying on unreliable pseudo-labels, VSFOT employs VLM-derived semantic priors and an OT-based matching strategy to achieve stable and reliable adaptation. To further enhance domain alignment, VSFOT incorporates a bidirectional distillation mechanism in which the model learns semantic consistency from the VLM, while the VLM is refined using task-specific cues from the model. These two stages alternate during training. By combining the generalization ability of the VLM with the discriminative power of the task model, VSFOT achieves robust, source-free adaptation and consistently outperforms existing SFDA methods on four benchmark datasets.



### Framework

![Framework](images/fig2_framework.png)



### Environment

The code is tested with Python 3.x and PyTorch. Install the dependencies with: `pip install -r requirements.txt`


### Dataset

This project supports the following datasets:

| Dataset argument | Domains |
| --- | --- |
| `OFFICE31` | `amazon`, `dslr`, `webcam` |
| `OfficeHomeDataset` | `art`, `clipart`, `product`, `realworld` |
| `VisDA` | `synthesis`, `real` |
| `DomainNet-126` | `clipart`, `painting`, `real`, `sketch` |

The repository provides image-list files under `dataset/<dataset_name>/*.txt`.

Before running the code, modify `IMAGEFILE` at the beginning of both `main.py` and `trainsSurce.py` to the root path of the raw datasets. Make sure the raw image folders under `IMAGEFILE` match the paths recorded in the corresponding `.txt` files under `dataset/`.

Default paths in the code: `IMAGEFILE = "/root/user/dataset"`


### Usage

The training procedure has two stages:

1. Train the source model with labeled source-domain data.
2. Run source-free adaptation on the unlabeled target-domain data.

**1. Train Source Model**
```bash
python trainSource.py --name Source\Source_realworld --dataset OfficeHomeDataset --source realworld
python trainSource.py --name Source\Source_product --dataset OfficeHomeDataset --source product
python trainSource.py --name Source\Source_clipart --dataset OfficeHomeDataset --source clipart
python trainSource.py --name Source\Source_art --dataset OfficeHomeDataset --source art
```

The source checkpoint will be saved as: `process/Source/Source_<domain>/Source.pt`



**2. Adapt to Target Domain**

```bash
python main.py --device 0 --name OfficeHome/Ar2Cl --epochs 40 --dataset OfficeHomeDataset --source art --target clipart
python main.py --device 0 --name OfficeHome/Ar2Pr --epochs 40 --dataset OfficeHomeDataset --source art --target product
python main.py --device 0 --name OfficeHome/Ar2Re --epochs 40 --dataset OfficeHomeDataset --source art --target realworld
python main.py --device 0 --name OfficeHome/Cl2Ar --epochs 40 --dataset OfficeHomeDataset --source clipart --target art
python main.py --device 0 --name OfficeHome/Cl2Pr --epochs 40 --dataset OfficeHomeDataset --source clipart --target product
python main.py --device 0 --name OfficeHome/Cl2Re --epochs 40 --dataset OfficeHomeDataset --source clipart --target realworld
python main.py --device 0 --name OfficeHome/Pr2Ar --epochs 40 --dataset OfficeHomeDataset --source product --target art
python main.py --device 0 --name OfficeHome/Re2Ar --epochs 40 --dataset OfficeHomeDataset --source realworld --target art
python main.py --device 0 --name OfficeHome/Pr2Cl --epochs 40 --dataset OfficeHomeDataset --source product --target clipart
python main.py --device 0 --name OfficeHome/Pr2Re --epochs 40 --dataset OfficeHomeDataset --source product --target realworld
python main.py --device 0 --name OfficeHome/Re2Cl --epochs 40 --dataset OfficeHomeDataset --source realworld --target clipart
python main.py --device 0 --name OfficeHome/Re2Pr --epochs 40 --dataset OfficeHomeDataset --source realworld --target product

```

Outputs, logs, checkpoints, and copied code snapshots are saved under: `process/<experiment_name>/`


### Additional Note

1. For weight information and validation details, please refer to the `model` branch.

2. Email: sureshan@foxmail.com
