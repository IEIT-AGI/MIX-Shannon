#!/bin/bash

experiment_name=FCTR_CopsRef
experiment_log_name=${experiment_name}.out
echo "training $experiment_name "


CUDA_VISIBLE_DEVICES=0,1,2,3   \
nohup python train.py \
--config-name train_fctr_e_further_copsRef  \
trainer.gpus=4 \
+trainer.strategy=ddp \
trainer.gradient_clip_val=0.1 \
trainer.max_epochs=20 \
trainer.check_val_every_n_epoch=21 \
> ${experiment_log_name} 2>&1 &

