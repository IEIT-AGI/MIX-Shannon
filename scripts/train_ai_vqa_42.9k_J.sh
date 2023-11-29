#!/bin/bash

experiment_name=ai_vqa_429k
experiment_log_name=${experiment_name}.out
echo "training $experiment_name "


CUDA_VISIBLE_DEVICES=4,5,6,7   \
nohup python train.py \
--config-name train_ai_vqa_42.9k_J  \
trainer.gpus=4 \
+trainer.strategy=ddp \
trainer.gradient_clip_val=2 \
trainer.max_epochs=35 \
> ${experiment_log_name} 2>&1 &