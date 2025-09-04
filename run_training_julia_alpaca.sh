#!/bin/bash

# Training Script for Julia Combined Dataset (ExAi) - Fixed CUDA paths

set -e

# Configuration
SCRIPT_NAME="finetune_julia.py"
DATA_PATH="../../../scratch/shared_dir/alpaca_datasets/alpaca_julia.jsonl"
OUTPUT_PATH="../../../scratch/shared_dir/ian6/finetuned_models/julia"
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"
NUM_GPUS=2

echo "🚀 Starting Julia Combined Dataset Training..."
echo "📊 Model: $MODEL_PATH"
echo "📁 Data: $DATA_PATH"  
echo "💾 Output: $OUTPUT_PATH"
echo "🔧 GPUs: $NUM_GPUS"
echo "📈 Total samples: 124,101"

# Fix CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export PYTHONPATH=$PWD:$PYTHONPATH

# Verify CUDA setup
echo "🔍 Verifying CUDA setup..."
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC version:"
nvcc --version

# Create output directory structure
mkdir -p $OUTPUT_PATH
mkdir -p $OUTPUT_PATH/logs

echo "🎯 Starting training at $(date)..."

# Run distributed training with optimized parameters for the dataset size
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    $SCRIPT_NAME \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --logging_steps 25 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --max_steps 1500 \
    --report_to "wandb" \
    --run_name "julia-alpaca-$(date +%Y%m%d_%H%M%S)" \
    --logging_dir "$OUTPUT_PATH/logs" \
    2>&1 | tee $OUTPUT_PATH/training.log

echo "✅ Training completed at $(date)"
echo "📋 Logs saved to: $OUTPUT_PATH/training.log"
echo "💾 Model saved to: $OUTPUT_PATH"
echo "🎉 Finetuned model ready for use!"
