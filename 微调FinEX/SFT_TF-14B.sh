# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
# CUDA_VISIBLE_DEVICES=1,0 python src/train_bash.py \

# 将部分数据offload到CPU，降低对显存的需求（作用不大）
# deepspeed --num_gpus=1 src/train_bash.py \

# 加速不减存
# deepspeed --num_gpus=2  src/train_bash.py --deepspeed script/ds_configs/ds_config_2.json \
# deepspeed --num_gpus=2  src/train_bash.py --deepspeed script/ds_configs/ds_config_st3.json \
# accelerate launch src/train_bash.py \


deepspeed --num_gpus=2  src/train_bash.py --deepspeed script/ds_configs/ds_config_st3.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/QFRC/ZQS/Projects/LLM_models/Tongyi-Finance-14B \
    --dataset feature_mix_80 \
    --cutoff_len 18000 \
    --template default \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir /home/QFRC/ZQS/Projects/LLaMA-Factory-main/script/TF-14B/Exports/export_event80_175_18000_save \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate 5e-5 \
    --num_train_epochs 475 \
    --plot_loss \
    --fp16