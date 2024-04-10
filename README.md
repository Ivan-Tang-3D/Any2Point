# Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding
Official implementation of ['Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding']().

<p align="center">                                                                                                                                          <img src=""/ width="45%"> <br>
</p>

## Introduction


<div align="center">
  <img src=""/>
</div>

## Main Results
We report the pre-training modality (Pre-train), the number of learnable parameters (\#Param) on the "PB-T50-RS" split of ScanObjectNN (SCAN.) and ModelNet40 (MN.).{$^\dagger$} indicates utilizing the voting strategy.
| Method                  | Pre-train  | #Param(M) | SCAN.(%) | MN.(%)   |
|-------------------------|------------|-----------|----------|----------|
| PointNet                | N/A        | 3.5       | 68.0     | 89.2     |
| PointNet++              | N/A        | 1.5       | 77.9     | 90.7     |
| DGCNN                   | N/A        | 1.8       | 78.1     | 92.9     |
| PointMLP                | N/A        | 12.6      | 85.4     | 94.1     |
| Point-PN                | N/A        | 0.8       | 87.1     | 93.8     |
| PointNeXt               | N/A        | 1.4       | 87.7     | 94.0     |
| Point-BERT              | 3D         | 22.1      | 83.1     | 92.7     |
| Point-MAE               | 3D         | 22.1      | 85.2     | 93.2     |
| Point-M2AE              | 3D         | 15.3      | 86.4     | 93.4     |
| P2P-HorNet              | 2D         | 1.2       | 89.3     | 94.0$^\dagger$ |
| ACT                     | 3D+2D      | 22.1      | 88.2     | 93.7     |
| I2P-MAE                 | 3D+2D      | 12.9      | 90.1     | 93.7     |
| ReCon                   | 3D+2D+Language | 43.6  | 90.6     | 94.1     |
| Any2Point (Audio)       | Audio      | **0.8**   | **87.0** | **92.7** |
| Any2Point (2D)          | 2D         | **0.8**   | **87.7** | **93.2** |
| Any2Point (Language)    | Language   | **0.9**   | **91.9** | **94.3** |



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
