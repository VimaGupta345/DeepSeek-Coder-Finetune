#!/usr/bin/env python

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import transformers
import copy

# Configuration
checkpoint_path = "/scratch/shared_dir/finetuned_models/output/full_training_rust_base_20250626_182250"
hf_dataset_name = "Maverfrick/Rust_dataset"
hf_eval_split = "train[100000:110000]"

print(f"🔍 Evaluating model from: {checkpoint_path}")
print(f"📊 Dataset: {hf_dataset_name}")
print(f"📋 Eval split: {hf_eval_split}")

# Load tokenizer and model with CPU fallback
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path, 
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu"  # Use CPU to avoid CUDA issues
)

# Load evaluation dataset
print(f"Loading evaluation dataset: {hf_dataset_name} split {hf_eval_split}")
eval_raw_dataset = load_dataset(hf_dataset_name, split=hf_eval_split)

# Take a smaller sample for quick evaluation
eval_sample = eval_raw_dataset.select(range(min(1000, len(eval_raw_dataset))))
print(f"Evaluation sample size: {len(eval_sample)}")

# Tokenization function (same as training)
def build_instruction_prompt(instruction: str) -> str:
    return f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:
"""

def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = tokenizer(examples, max_length=512, truncation=True, padding=False)  # Shorter for eval
    sources_tokenized = tokenizer(sources, max_length=512, truncation=True, padding=False)
    
    input_ids_labels = []
    for tokenized_full, tokenized_s in zip(examples_tokenized["input_ids"], sources_tokenized["input_ids"]):
        input_ids_label = copy.deepcopy(tokenized_full)
        for i in range(len(tokenized_s)):
            input_ids_label[i] = -100
        input_ids_labels.append(input_ids_label)
    
    return dict(input_ids=examples_tokenized["input_ids"], labels=input_ids_labels)

def eval_tokenize_function(examples, tokenizer):
    EOT_TOKEN = "<|EOT|>"
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['response']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

# Tokenize evaluation dataset
print("Tokenizing evaluation dataset...")
tokenized_eval_dataset = eval_sample.map(
    eval_tokenize_function,
    batched=True,
    batch_size=100,
    num_proc=8,
    remove_columns=eval_sample.column_names,
    desc="Tokenizing Eval Set",
    fn_kwargs={"tokenizer": tokenizer}
)

print(f"Tokenized evaluation dataset size: {len(tokenized_eval_dataset)}")

# Data collator
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# Setup evaluation with minimal resources
training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=1,  # Very small batch size
    dataloader_pin_memory=False,
    bf16=False,  # Disable bf16 for CPU
    fp16=False,  # Disable fp16 for CPU
    use_cpu=True,  # Force CPU usage
    report_to=[],  # Disable wandb for evaluation
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Run evaluation
print("🚀 Starting evaluation...")
try:
    eval_results = trainer.evaluate()
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n✅ Evaluation completed!")
    print(f"💾 Results saved to: ./eval_results")
    
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    print("Let's try a simpler manual evaluation...")
    
    # Manual evaluation
    print("\n🔧 Running manual evaluation...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(tokenized_eval_dataset):
            if i >= 10:  # Just evaluate first 10 samples
                break
                
            input_ids = torch.tensor([sample['input_ids']])
            labels = torch.tensor([sample['labels']])
            
            # Calculate loss manually
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
            
            if i % 5 == 0:
                print(f"Sample {i+1}: loss = {loss.item():.4f}")
    
    avg_loss = total_loss / min(10, len(tokenized_eval_dataset))
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n📊 Manual Evaluation Results (sample of 10):")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
