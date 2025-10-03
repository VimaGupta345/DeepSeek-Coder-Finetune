#!/bin/bash

# Direct H200 Training Script (without tmux)
# Usage: ./run_training_direct.sh

set -e

# Configuration
SCRIPT_NAME="finetune_full_deepseekcoder_includes_eval_rustcontinue.py"
DATA_PATH="data/rust_instruct_format2.jsonl"
# $(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="../../../scratch/shared_dir/finetuned_models/output/full_training_rust_20250626_130748"
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"
NUM_GPUS=3

echo "🚀 Starting H200 6-GPU training directly..."
echo "📊 Model: $MODEL_PATH"
echo "📁 Data: $DATA_PATH"  
echo "💾 Output: $OUTPUT_PATH"
echo "🔧 GPUs: $NUM_GPUS"

# Set environment variables
export CUDA_VISIBLE_DEVICES=4,5,6
export NCCL_DEBUG=INFO
export PYTHONPATH=$PWD:$PYTHONPATH

# Create output directory
mkdir -p $OUTPUT_PATH
mkdir -p $OUTPUT_PATH/logs

echo "🎯 Starting training at $(date)..."

# Run distributed training with logging
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    $SCRIPT_NAME \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 50 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --max_steps 1500 \
    --report_to "wandb" \
    --run_name "h200-deepseek-full-20250626_130748" \
    --logging_dir "$OUTPUT_PATH/logs" \
    --resume_from_checkpoint "$OUTPUT_PATH/checkpoint-1000" \
    2>&1 | tee $OUTPUT_PATH/training.log

echo "✅ Training completed at $(date)"
echo "📋 Logs saved to: $OUTPUT_PATH/training.log"
echo "💾 Model saved to: $OUTPUT_PATH"
