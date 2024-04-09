# Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding
Official implementation of ['Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding']().

<p align="center">                                                                                                                                          <img src=""/ width="45%"> <br>
</p>

## Introduction

We propose the Point-PEFT, a novel framework for adapting point cloud pre-trained models with minimal learnable parameters. Specifically, for a pre-trained 3D model, we freeze most of its parameters, and only tune the newly added PEFT modules on downstream tasks, which consist of a Point-prior Prompt and a Geometry-aware Adapter. The Point-prior Prompt constructs a memory bank with domain-specific knowledge and utilizes a parameter-free attention for prompt enhancement. The Geometry-aware Adapter aims to aggregate point cloud features within spatial neighborhoods to capture fine-grained geometric information.

<div align="center">
  <img src=""/>
</div>

## Main Results
Comparison with existing 3D pre-trained models on the PB-T50-RS split of ScanObjectNN:
| Method | Parameters | PB-T50-RS|
| :-----: | :-----: |:-----:|
| [Point-BERT](https://github.com/lulutang0608/Point-BERT) | 22.1M | 83.1%| 
| **+Point-PEFT** | **0.6M** | **85.0%**|
| [Point-MAE-aug](https://github.com/Pang-Yatian/Point-MAE) | 22.1M | 88.1%|
| **+Point-PEFT** | **0.7M** | **89.1%**|
| [Point-M2AE-aug](https://github.com/ZrrSkywalker/Point-M2AE)| 12.9M | 88.1%|
| **+Point-PEFT** | **0.7M** | **88.2%**|

Comparison with existing 3D pre-trained models on the ModelNet40 without voting method:
| Method | Parameters | Acc|
| :-----: | :-----: |:-----:|
| [Point-BERT](https://github.com/lulutang0608/Point-BERT) | 22.1M | 92.7%| 
| **+Point-PEFT** | **0.6M** | **93.4%**|
| [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) | 22.1M | 93.2%|
| **+Point-PEFT** | **0.8M** | **94.2%**|
| [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE)| 15.3M | 93.4%|
| **+Point-PEFT** | **0.6M** | **94.1%**|


## Ckpt Release

Real-world shape classification on the PB-T50-RS split of ScanObjectNN:
| Method | Config | Acc.| Logs |
| :-----: | :-----:|:-----:| :-----:|
| Point-M2AE-aug | [scan.yaml](https://drive.google.com/file/d/1JzYQKMTGLmT4cQ3HNI9g8rNHFovmPMgl/view?usp=sharing) | 88.2% | [scan_m2ae.log](https://drive.google.com/file/d/1Dx8ucp_7_2GtSe60wq3jsbtn4xUKHqM8/view?usp=sharing) |


## Get Started

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/EvenJoker/Point-PEFT.git
cd Point-PEFT

conda create -n point-peft python=3.8
conda activate point-peft

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
# e.g., conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

pip install -r requirements.txt
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
### Dataset
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially Put the unzip folder under `data/`.

The final directory structure should be:
```
│Point-PEFT/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ScanObjectNN/
├──...
```

### Fine-tuning
Please download the [ckpt-best.pth](https://drive.google.com/file/d/16oJrxbLlDLMp1nA8W3EEjRA-cENReAU9/view?usp=sharing), [pre-train.pth](https://drive.google.com/file/d/1m9biTvZN098NP3IwJuTt3kWI0t-sIKSn/view?usp=sharing) and [cache_shape.pt](https://drive.google.com/file/d/1YdUlBL2QpimMBvyK3XaDcUCVxMQP1-1h/view?usp=sharing) into the `ckpts/` folder. 

For the PB-T50-RS split of ScanObjectNN, run:
```bash
sh Finetune_cache_prompt_scan.sh
```

## Acknowledgement
This repo benefits from [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE). Thanks for their wonderful works.
