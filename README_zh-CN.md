<div align="center">

# MIX-Shannon
[English](README.md) | 简体中文
</div>

<br>

## ✈️ 简介

**MIX-Shannon** 是一个探索多模态领域中新任务的项目，例如进一步的指代表达理解（**FREC**），代理互动视觉问答（**AI-VQA**），烹饪步骤图像检索（**CSI-Recipe**），以及图像视觉问答（**VQAI**）等.
<br>

## 🚀 安装
**此部分将演示如何配置工程.**

分为以下三个步骤:
1. 克隆此项目:
    
    ```bash
    git clone https://github.com/IEIT-AGI/MIX-Shannon.git
    cd MIX-Shannon
    ```
    
2. (可选) 基于conda创建环境并激活:
    
    ```bash
    conda create -n MIX-Shannon python=3.8
    conda activate MIX-Shannon
    ```
    
3. 安装依赖包:
    
    ```bash
    pip install -r requirements.txt
    ```
    

使用`requirements.txt`一次安装所有依赖包

<br>

## ⚡ 已支持的任务
| Task                                        | Dataset                   | data and code                                      |
|---------------------------------------------|---------------------------|----------------------------------------------------|
| Further Referring Expression Comprehension  | RefCOCOs CopsRef Talk2Car | [FREC](projects/FREC/fctr_frec_zh.md)              |
| Agent Interaction Visual Question Answering | AI-VQA                    | [AI-VQA](projects/AI-VQA/ai-vqa_zh.md)             |
| Cooking-step-images Retrieval               | CSIR                      | [CSI-Recipe](projects/CSI-Recipe/csi_recipe_zh.md) |
| visual question answering with image        | VQAI                      | [VQAI](projects/VQAI/lgd_vqai_zh.md)               |





<br>

## 🙏 致谢
此工程搭建过程中主要参考以下两个项目.

特别感谢:
+ [**Lightning-hydra-template**](https://github.com/ashleve/lightning-hydra-template)
+ [**MMEngine**](https://github.com/open-mmlab/mmengine)

<br>

## ☎️ 联系我们
如何您在使用过程中有任何问题、建议，请联系我们[ieitagi001@gmail.com](ieitagi001@gmail.com).


<br>

## 📄 证书
该项目采用 [Apache 2.0 license](resources/LICENSE).
