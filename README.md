<div align="center">

# MIX-Shannon
English | [简体中文](README_zh-CN.md)
</div>

<br>

## ✈️ Introduction

**MIX-Shannon** is a project that explores new tasks in the field of multi-modalities, such as Further  Referring Expression Comprehension (**FREC**), Agent Interaction Visual Question Answering (**AI-VQA**), Cooking-step-images Retrieval  (**CSI-Recipe**),and visual question answering with image (**VQAI**).

<br>

## 🚀 Installation
**In this section, we demonstrate how to set up  environment for our project.**

To get started, follow these steps:
1. Clone the project repository:
    
    ```bash
    git clone https://github.com/IEIT-AGI/MIX-Shannon.git
    cd MIX-Shannon
    ```
    
2. (Optional) Create a conda environment and activate it:
    
    ```bash
    conda create -n MIX-Shannon python=3.8
    conda activate MIX-Shannon
    ```
    
3. Install the required packages:
    
    ```bash
    pip install -r requirements.txt
    ```
    

We  have a `requirements.txt` file that you can use to install the required packages all at once.

<br>

## ⚡ Supported Tasks
| Task                                        | Dataset                   | data and code                                   |
|---------------------------------------------|---------------------------|-------------------------------------------------|
| Further Referring Expression Comprehension  | RefCOCOs CopsRef Talk2Car | [FREC](projects/FREC/fctr_frec.md)              |
| Agent Interaction Visual Question Answering | AI-VQA                    | [AI-VQA](projects/AI-VQA/ai-vqa.md)             |
| Cooking-step-images Retrieval               | CSIR                      | [CSI-Recipe](projects/CSI-Recipe/csi_recipe.md) |
| visual question answering with image        | VQAI                      | [VQAI](projects/VQAI/lgd_vqai.md)               |

<br>

## 🙏 Credits
Credits and sources are provided throughout this repo.

Special thanks to:
+ [**Lightning-hydra-template**](https://github.com/ashleve/lightning-hydra-template)
+ [**MMEngine**](https://github.com/open-mmlab/mmengine)

<br>

## ☎️ Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at [ieitagi001@gmail.com](ieitagi001@gmail.com)..

<br>

## 📄 License
This project is released under the [Apache 2.0 license](resources/LICENSE).
