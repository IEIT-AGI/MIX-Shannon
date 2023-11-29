<div align="center">

# Heterogeneous Cross-Modal Recipe Retrieval on Cooking Step Images

English | [简体中文](csi_recipe_zh.md)
</div>

<br>

In this project, we propose a retrieval framework based on temporal step images, we design Attention Embedding Networks and BiLSTM to capture time series information, and also use Heterogeneous Graph Neural Networks to fuse multi-structured textual information with good results.

<br>

## Introduction

![CSIR-task](../resources/CSIR-task.jpg)

Cooking-step-images retrieval (**CSIR**) refers to the problem of retrieving procedural images from a list of cooking step candidate images given a textual recipe as the query, or the reverse side ,as shown in the figure above . To facilitate further development of this task, we have open-sourced the data, models, and code.

1. This code aims to achieve cross-modal recipe retrieval based on cooking step images.
2. We have collected a new dataset called CSI-Recipe from several recipe websites. It contains 12,330 recipes and 109,330 cooking-step images. Each recipe includes cooking step images, ingredients, instructions, and titles, with text in both Chinese and English.
3. We propose a baseline method for our recipe retrieval task by introducing a heterogeneity graph. Our ingredients-instruction heterogeneity graph can dynamically enhance the key instruction semantics.


## Usage
### Requirements
You need to install `dgl` before using it.
```bash
    pip install dgl
```


<br>

## Getting Started
### Data Preparation

The CSI-Recipe task involves two input modalities: images and text. To extract the corresponding features, we employ visual backbone networks and text backbone networks. Specifically, we use ResNet50, ResNeXt101, and ViT to extract image features, and Transform to extract text features. To accelerate the training process, we pre-extract the image features and provide download links for them.

**Baidu NetDisk**: [CSI-Recipe features download](https://pan.baidu.com/s/1c249Nbr2IdvvHDjbrT0srA?pwd=miys) Extract code: miys 

**Google Drive**: [CSI-Recipe features download](https://drive.google.com/file/d/1YdP3jO0Qs1-SEXH6Hp15akTTeAlrMI8Y/view?usp=sharing)

The dataset is currently subject to copyright, please contact us ([wang.lilc@ieisystem.com](wang.lilc@ieisystem.com)) to sign a copyright agreement and we will send a download link via email.

### How to train
Train model with default configuration:（ ResNet50 visual feature and Roberta text feature）

```bash
CUDA_VISIBLE_DEVICES=0  python train.py --config-name train_recipe name=Roberta_ResNet50 trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
```

Train model with chose different backbone networks from [configs/datamodule](/configs/datamodule)

```bash
CUDA_VISIBLE_DEVICES=0    python train.py --config-name train_recipe_robeta_vitbase  name=Roberta_vitbase trainer.gpus=1 trainer.gradient_clip_val=1  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=0    python train.py --config-name train_recipe_bertlarge_resnet50  name=BERTlarge_ResNet50 trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=0    python train.py --config-name train_recipe_robeta_resnext101 name=Roberta_ResNeXt101 trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=0    python train.py --config-name train_recipe_albert_resnet50  name=albert_ResNet50 trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
```
<br>

## Results and Models
All experiments  were trained on a single GPU, and the results are as follows.

![experiments](../resources/CSIR-experiments.jpg)

## Credits
The **GAT**(Graph Attention Network) code is sourced from **[HeterSumGraph](https://github.com/dqwang122/HeterSumGraph)**. Thanks for their work.
## Citation
If you use this toolbox or benchmark in your research, please cite this project.
