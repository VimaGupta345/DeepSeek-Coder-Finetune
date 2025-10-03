#!/bin/bash

# Direct H200 Training Script for Ruby (without tmux)
# Usage: ./run_training_ruby_direct.sh

set -e

# Configuration
SCRIPT_NAME="finetune_full_deepseekcoder_ruby_includes_eval.py"
DATA_PATH="data/ruby_instruct_format.jsonl"
OUTPUT_PATH="../../../scratch/shared_dir/finetuned_models/output/full_training_ruby_instruct_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"
NUM_GPUS=4

echo "🚀 Starting H200 7-GPU Ruby training directly..."
echo "📊 Model: $MODEL_PATH"
echo "📁 Data: $DATA_PATH"  
echo "💾 Output: $OUTPUT_PATH"
echo "🔧 GPUs: $NUM_GPUS"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,6,7
export NCCL_DEBUG=INFO
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=32

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
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 4 \
    --learning_rate 1.5e-5 \
    --warmup_steps 10 \
    --logging_steps 100 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --max_steps 1500 \
    --report_to "wandb" \
    --run_name "h200-deepseek-ruby-instructional-csn-torch$(date +%Y%m%d_%H%M%S)" \
    --logging_dir "$OUTPUT_PATH/logs" \
    2>&1 | tee $OUTPUT_PATH/training.log

echo "✅ Training completed at $(date)"
echo "📋 Logs saved to: $OUTPUT_PATH/training.log"
echo "💾 Model saved to: $OUTPUT_PATH"
