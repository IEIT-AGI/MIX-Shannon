#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

experiment_name=VQAI
experiment_log_name=${experiment_name}.out
echo "training $experiment_name "


 CUDA_VISIBLE_DEVICES=4,5,6,7  \
 nohup python train.py \
 --config-name train_vqai  \
 trainer.gpus=4 \
 trainer.max_epochs=70 \
 > ${experiment_log_name} 2>&1 &
