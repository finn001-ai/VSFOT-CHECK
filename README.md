## Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport

> ### 本仓库使用说明（复现版）
>
> 所有命令都在**项目根目录**下运行（Jupyter 里在前面加 `!`）。
>
> | 步骤 | 命令 | 说明 |
> |---|---|---|
> | 1. 数据软链接（只需一次） | `bash scripts/link_data.sh` | 把云端数据链接到 `dataset/OfficeHomeDataset/`，含 `classname.txt` |
> | 2. 训练源模型 + 12 个任务 | `bash scripts/run.sh` | 正式训练，结果在 `runs/Source_home/`、`runs/OfficeHome/` |
> | 3. OT 消融 | `bash scripts/run_ablation.sh` | 结果打包到 `runs/ablation_results/<日期>.zip` |
> | 4. 近邻平滑实验 | `bash scripts/run_knn.sh` | `--knn_k` / `--knn_mgva`，和消融基线一起汇总到 `runs/ablation_results/<日期>.zip` |
> | 5. 强视图对齐实验 | `bash scripts/run_strong.sh` | `--align_view strong/both`（Noisy Student），和消融基线一起汇总 |
> | 诊断：训练快照 + OT 指标 | `bash scripts/run_diag.sh` | 结果打包到 `runs/diag_results/<日期>.zip` |
> | 诊断：μ_p 消融等 | `bash scripts/run_diag_quick.sh` | 需要先跑上一行的快照 |
> | 诊断：μ_p 估计器对比 | `bash scripts/run_diag_mu.sh` | 需要先跑 `run_diag.sh` 的快照；结果打包到 `runs/diag_mu_results/<日期>.zip` |
> | 诊断：全数据集 OT | `bash scripts/run_diag_global.sh` | 需要先跑 `run_diag.sh` 的快照；结果打包到 `runs/diag_global_results/<日期>.zip` |
> | 诊断：目标域结构 | `bash scripts/run_diag_struct.sh` | 需要先跑 `run_diag.sh` 的快照；结果打包到 `runs/diag_struct_results/<日期>.zip` |
> | 诊断：簇级命名（方案 A） | `bash scripts/run_diag_cluster.sh` | 需要先跑 `run_diag.sh` 的快照；结果打包到 `runs/diag_cluster_results/<日期>.zip` |
> | 检查 CLIP 零样本准确率 | `python tools/zeroshot_clip.py` | 应为平均 79.04%（论文表 4） |
>
> - `scripts/`：运行脚本，配置都写在每个脚本开头，有注释。`tools/`：被脚本调用的汇总 / 诊断程序。
> - **所有输出都在 `runs/` 下**：训练产物 `runs/`，结果压缩包 `runs/<名称>_results/`，日志 `runs/logs/`。根目录只放代码。
> - 和公开代码的区别：Office-Home 类名改为从 `classname.txt` 读取（原写死的类名在第 58–63 类和标签错位）；
>   `main.py` 新增 `--align_mode`（OT 消融）、`--knn_k` / `--knn_mgva`（学生特征空间近邻平滑，默认关闭）、`--align_view`（强视图对齐，默认 weak）、`--snapshot_at` / `--stop_at`（诊断快照）、进度条和训练计时。
>   训练公式和超参与公开代码完全一致。修复类名后 Ar→Cl = 74.4%（论文 75.0%）。
> - 下方原 README 中 `--name Source\Source_art` 的反斜杠在 Linux 下会被吞掉，请用 `scripts/run.sh`，
>   它会按 `Source_home/Source_<域>` 命名（realworld 为 `Source_real world`）。
> - 实验记录在本地 `archive/`（不提交）。

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

The source checkpoint will be saved as: `runs/Source/Source_<domain>/Source.pt`



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

Outputs, logs, checkpoints, and copied code snapshots are saved under: `runs/<experiment_name>/`


### Additional Note

1. For weight information and validation details, please refer to the `model` branch.

2. Email: sureshan@foxmail.com
