# finally result
CUDA_VISIBLE_DEVICES=2    python train.py --config-name train_recipe_robeta_vitbase  name=robeta_vitbase-grad1_header_unuse_fc trainer.gpus=1 trainer.gradient_clip_val=1  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=3    python train.py --config-name train_recipe_bertlarge_resnet50  name=bertlarge_resnet50-grad1-nouser_fc trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=0  python train.py --config-name train_recipe_robeta_resnext101 name=robeta_resnext101-grad1_nousefc trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=2    python train.py --config-name train_recipe_albert_resnet50  name=albert_resnet50-grad1-nouse-fc trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4
CUDA_VISIBLE_DEVICES=0  python train.py --config-name train_recipe name=roberta_resNet50-grad1_nousefc trainer.gpus=1 trainer.gradient_clip_val=1.0  trainer.max_epochs=200 model.is_shuffle=False datamodule.num_workers=4


