#!/bin/bash

experiment_name=ai_vqa_81k-grd
experiment_log_name=${experiment_name}.out
echo "training $experiment_name "


CUDA_VISIBLE_DEVICES=0,1,2,3   \
nohup python train.py \
--config-name train_ai_vqa_8.1k_J.yaml  \
trainer.gpus=4 \
+trainer.strategy=ddp \
trainer.gradient_clip_val=2 \
trainer.max_epochs=35 \
> ${experiment_log_name} 2>&1 &