echo vqai-evaluator start running

ck_number=60
cuda_id=3

user_home=$HOME
ck_dir=$user_home/imix2.0_log/logs/experiments/runs/vqai-trainer/2023-09-09_04-10-02/checkpoints
output_dir=$user_home/imix2.0_log/vqai_generate_img/${ck_number}
ckpt_path="${ck_dir}/epoch_epoch\\=0${ck_number}.ckpt"

echo ck file: $ckpt_path
echo output_dir: $output_dir

CUDA_VISIBLE_DEVICES=${cuda_id}  \
nohup python validate.py --config-name eval_vqai.yaml \
trainer.gpus=1 \
model.evaluate.generate_img_dir=$output_dir  \
ckpt_path=$ckpt_path \
> vqai_eval_epoch_${ck_number}.log 2>&1 &