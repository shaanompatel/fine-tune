MODEL=meta-llama/Llama-3.2-1B
train_file=./data/formatted_dialog.txt
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
LEARNING_RATE=7e-5
TOTAL_BATCH_SIZE=1 # max 2 for 2 GPUs
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
OUTPUT_DIR=model_lr${LEARNING_RATE}_bs${TOTAL_BATCH_SIZE}
echo "Training ${MODEL} model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./config/stage3_no_offloading_accelerate.conf \
    ./finetune.py \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_slow_tokenizer \
    --train_file ${train_file} \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --output_dir ./models/${OUTPUT_DIR}_llama_march_7th_scratch/ \