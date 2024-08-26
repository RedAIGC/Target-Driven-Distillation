export TRAIN_SHARDS_PATH_OR_URL="/mnt/dataset/laion_6plus"
export PRETRAINED_TEACHER_MODEL="./stable-diffusion-xl-base-1.0"
export PRETRAINED_VAE_MODEL_NAME_OR_PATH="./sdxl-vae-fp16-fix"
accelerate launch --config_file=config.yaml train_tdd_adv.py \
    --pretrained_teacher_model=$PRETRAINED_TEACHER_MODEL \
    --pretrained_vae_model_name_or_path=$PRETRAINED_VAE_MODEL_NAME_OR_PATH \
    --train_shards_path_or_url=$TRAIN_SHARDS_PATH_OR_URL \
    --output_dir="result/TDD_uc0.2_etas0.3_ddim250_adv" \
    --seed=453645634 \
    --resolution=1024 \
    --max_train_samples=4000000 \
    --max_train_steps=100000 \
    --train_batch_size=14 \
    --dataloader_num_workers=32 \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=5000 \
    --validation_steps=500 \
    --learning_rate=2e-06 \
    --lora_rank=64 \
    --w_max=3.5 \
    --w_min=3.5 \
    --mixed_precision="fp16" \
    --loss_type="huber"  --use_fix_crop_and_size --adam_weight_decay=0.0 \
    --val_infer_step=4 \
    --gradient_checkpointing \
    --num_ddim_timesteps=250 \
    --proportion_empty_prompts=0.2 \
    --num_inference_steps_min=4 \
    --num_inference_steps_max=8 \
    --s_ratio=0.3 \
    --adv_lr=1e-5 \
    --adv_weight=0.1 \

# cd /mnt/nj-aigc/usr/polu
# bash run_gpu.sh