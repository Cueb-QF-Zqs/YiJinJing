python src/export_model.py \
    --model_name_or_path /home/QFRC/ZQS/Projects/LLaMA-Factory-main/script/TF-14B/Merges/TF-14B-Base2_epoch143_10G \
    --template default \
    --finetuning_type lora \
    --export_dir /home/QFRC/ZQS/Projects/LLaMA-Factory-main/script/TF-14B/Merges/TF-14B-Base2_epoch143_10G_q4 \
    --export_size 10 \
    --export_quantization_bit 4 \
    --export_quantization_dataset /home/QFRC/ZQS/Projects/LLaMA-Factory-main/data/Mine/q82.json \
    --export_legacy_format False