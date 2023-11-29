#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

experiment_name=CSI_Task_bertlarge_resnet50
experiment_log_name=${experiment_name}.out
echo "training $experiment_name "


CUDA_VISIBLE_DEVICES=0  \
nohup python train.py \
--config-name train_recipe_bertlarge_resnet50  \
trainer.gpus=1 \
seed=666 \
trainer.gradient_clip_val=2 \
trainer.max_epochs=200 \
> ${experiment_log_name} 2>&1 & 


