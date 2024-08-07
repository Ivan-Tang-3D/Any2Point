# Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding
Official implementation of ['Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding'](https://arxiv.org/pdf/2404.07989.pdf).

**[2023.5] We release ICCV2023 ['ViewRefer3D'](https://arxiv.org/pdf/2303.16894.pdf), a multi-view framework for 3D visual grounding exploring how to grasp the view knowledge from both text and 3D modalities with LLM.**

**[2023.9] We release AAAI2024 ['Point-PEFT'](https://arxiv.org/abs/2310.03059), adapting 3D pre-trained Models with 1% parameters to downstream tasks .**

**[2024.5] The results of Any2Point on ShapeNetPart will be released soon!**

**[2024.7] Any2Point has been accepted by ECCV 2024!**

<p align="center">                                                                                                                                          <img src="Teaser_any.png"/ width="70%"> <br>
</p>

## Introduction
Large foundation models have recently emerged as a prominent focus of interest, attaining superior performance in widespread scenarios. Due to the scarcity of 3D data, many efforts have been made to adapt pre-trained transformers from vision to 3D domains. However, such 2D-to-3D approaches are still limited, due to the potential loss of spatial geometries and high computation cost. More importantly, their frameworks are mainly designed for 2D models, lacking a general any-to-3D paradigm. In this paper, we introduce **Any2Point**, a parameter-efficient method to empower any-modality large models (vision, language, audio) for 3D understanding. Given a frozen transformer from any source modality, we propose a 3D-to-any (1D or 2D) virtual projection strategy that correlates the input 3D points to the original 1D or 2D positions within the source modality. This mechanism enables us to assign each 3D token with a positional encoding paired with the pre-trained model, which avoids 3D geometry loss caused by the true projection and better motivates the transformer for 3D learning with 1D/2D positional priors. Then, within each transformer block, we insert an any-to-3D guided adapter module for parameter-efficient fine-tuning. The adapter incorporates prior spatial knowledge from the source modality to guide the local feature aggregation of 3D tokens, compelling the semantic adaption of any-modality transformers. We conduct extensive experiments to showcase the effectiveness and efficiency of our method.

<div align="center">
  <img src="Intro_any.png"/>
</div>

## Main Results
We report the pre-training modality (Pre-train), the number of learnable parameters (\#Param) on the "PB-T50-RS" split of ScanObjectNN (SCAN.) and ModelNet40 (MN.). * indicates utilizing the voting strategy.
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
| P2P-HorNet              | 2D         | 1.2       | 89.3     | 94.0*    |
| ACT                     | 3D+2D      | 22.1      | 88.2     | 93.7     |
| I2P-MAE                 | 3D+2D      | 12.9      | 90.1     | 93.7     |
| ReCon                   | 3D+2D+Language | 43.6  | 90.6     | 94.1     |
| Any2Point (Audio)       | Audio      | **0.8**   | **87.0** | **92.7** |
| Any2Point (2D)          | 2D         | **0.8**   | **87.7** | **93.2** |
| Any2Point (Language)    | Language   | **0.9**   | **91.9** | **94.3** |



## Ckpt Release

Real-world shape classification on the PB-T50-RS split of ScanObjectNN:
| Method | Logs | Acc.| Ckpts |
| :-----: | :-----:|:-----:| :-----:|
| Any2Point-Lang-CLIP | [Language_CLIP_Scan.log](https://drive.google.com/file/d/1NGBY87PiTf8mkAThRjJCiUsSidRv28Kb/view?usp=sharing) | 91.9% | [Language_CLIP_Scan.pth](https://drive.google.com/file/d/1votPHTcD3KQ6fmdcZvUCZc5yMq4xbK0V/view?usp=sharing) |
| Any2Point-Vision-DINOV2 | [Vision_DINOV2_Scan.log](https://drive.google.com/file/d/1apIv0AOlL4utn7lXofhX_Tv2xy0EvAS_/view?usp=sharing) | 87.7% | [Vision_DINOV2_Scan.pth](https://drive.google.com/file/d/1cTCKNc5xfbMvLnQOj9BQf-uHg3Fime-W/view?usp=sharing) |
| Any2Point-Audio-ImageBind | [Audio_imagebind_scan.log](https://drive.google.com/file/d/176DtBKcWyn1Y22w2X0vQ0qz0sYqBYofv/view?usp=sharing) | 87.0% | [Audio_imagebind_scan.pth](https://drive.google.com/file/d/1mKFeKVfSyiBba2DNEY6gPPqDrINQBS2F/view?usp=sharing) |

Synthetic shape classification on the ModelNet40:
| Method | Logs | Acc.| Ckpts |
| :-----: | :-----:|:-----:| :-----:|
| Any2Point-Lang-CLIP | [Language_CLIP_ModelNet.log](https://drive.google.com/file/d/1-yR0E9iN2XmMbqFylxdEeIRfYZu2DlAp/view?usp=sharing) | 94.3% | [Language_CLIP_ModelNet.pth](https://drive.google.com/file/d/1DCIA6gfCjwOmauo7m4gh03DxOQB5KG4d/view?usp=sharing) |
| Any2Point-Vision-DINOV2 | [Vision_DINOV2_ModelNet.log](https://drive.google.com/file/d/122LQF8PkF8HiqkzP9OTdT34OrJmK4aKs/view?usp=sharing) | 93.2% | [Vision_DINOV2_ModelNet.pth](https://drive.google.com/file/d/1bApmj6GgCPcJrlm7Ke3pFs34LApGUnjX/view?usp=sharing) |
| Any2Point-Audio-ImageBind | [Audio_imagebind_ModelNet.log](https://drive.google.com/file/d/1UBppIkS65TdgpCB-w6jju-atZ5Tqup3v/view?usp=sharing) | 92.7% | [Audio_imagebind_ModelNet.pth](https://drive.google.com/file/d/163-Pskgx8GXdlN26itbds8fqmfzTYMl2/view?usp=sharing) |


## Get Started

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/Ivan-Tang-3D/Any2Point.git
cd Any2Point

conda create -n Any2Point python=3.7
conda activate Any2Point

# Install the according versions of torch and torchvision
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch

conda install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y
pip install torch-geometric==2.0

source install.sh
```

### Dataset
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially Put the unzip folder under `data/`.
The Language Part Training just occupies 26GB Memory.

The final directory structure should be:
```
│Any2Point/
├──Any2Point_CLIP_Lang/
├──ckpts/
├──data/
│   ├──ModelNet/
│   ├──ScanObjectNN/
├──...
```

### Fine-tuning
Please download the [CLIP_pre-train.pth](https://drive.google.com/file/d/1ok_f68lazKE-tcy_x_oJhhV58VdJnDXz/view?usp=sharing), [DINOV2_pre-train.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) and [ImageBind_audio_pre-train.pth](https://drive.google.com/file/d/1TvpfMhKcYKdPNrQzRP0KVpXUm2LPiGtH/view?usp=sharing) into the `ckpts/` folder. 

For the PB-T50-RS split of ScanObjectNN, run:

Any2Point_CLIP_Lang
```bash
cd Any2Point_CLIP_Lang
sh fine_tune.sh
```
Any2Point_DINOV2_Vision
```bash
cd Any2Point_DINOV2_Vision
sh fine_tune.sh
```
Any2Point_ImageBind_audio
```bash
cd Any2Point_ImageBind_audio
sh fine_tune.sh
```
For the ModelNet40, run:

Any2Point_CLIP_Lang
```bash
cd Any2Point_clip_lang_modelnet
sh fine_tune.sh
```
Any2Point_DINOV2
```bash
cd Any2Point_DINOV2_modelnet
sh fine_tune.sh
```
Any2Point_ImageBind
```bash
cd Any2Point_ImageBind_Modelnet
sh fine_tune.sh
```

### Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝.
```bash
@article{tang2024any2point,
  title={Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding},
  author={Tang, Yiwen and Liu, Jiaming and Wang, Dong and Wang, Zhigang and Zhang, Shanghang and Zhao, Bin and Li, Xuelong},
  journal={arXiv preprint arXiv:2404.07989},
  year={2024}
}
```

## Acknowledgement
This repo benefits from [Pix4Point](https://github.com/guochengqian/Pix4Point), [Point-NN](https://github.com/ZrrSkywalker/Point-NN), [PointTransformerV2](https://github.com/Pointcept/PointTransformerV2), [Openpoints](https://github.com/guochengqian/openpoints). Thanks for their wonderful works.
