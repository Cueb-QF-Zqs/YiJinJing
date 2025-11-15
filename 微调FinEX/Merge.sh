python src/export_model.py \
    --model_name_or_path /home/QFRC/ZQS/Projects/LLM_models/Tongyi-Finance-14B \
    --adapter_name_or_path /home/QFRC/ZQS/Projects/LLaMA-Factory-main/script/TF-14B/Exports/export_event80_175_18000_save/checkpoint-900 \
    --template default \
    --finetuning_type lora \
    --export_dir /home/QFRC/ZQS/Projects/LLaMA-Factory-main/script/TF-14B/Merges/TF-14B-80event_cp900 \
    --export_size 10 \
    --export_legacy_format False