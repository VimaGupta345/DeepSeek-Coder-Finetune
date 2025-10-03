#!/usr/bin/env python

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import transformers

# Configuration
checkpoint_path = "/scratch/shared_dir/finetuned_models/output/full_training_rust_base_20250626_182250"
hf_dataset_name = "Maverfrick/Rust_dataset"
hf_eval_split = "train[100000:110000]"

print(f"🔍 Evaluating model from: {checkpoint_path}")
print(f"📊 Dataset: {hf_dataset_name}")
print(f"📋 Eval split: {hf_eval_split}")

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path, 
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load evaluation dataset (same split as training)
print(f"Loading evaluation dataset: {hf_dataset_name} split {hf_eval_split}")
eval_raw_dataset = load_dataset(hf_dataset_name, split=hf_eval_split)

print(f"Evaluation dataset size: {len(eval_raw_dataset)}")

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
    examples_tokenized = tokenizer(examples, max_length=1024, truncation=True, padding=False)
    sources_tokenized = tokenizer(sources, max_length=1024, truncation=True, padding=False)
    
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
import copy

tokenized_eval_dataset = eval_raw_dataset.map(
    eval_tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=16,
    remove_columns=eval_raw_dataset.column_names,
    desc="Tokenizing Eval Set",
    fn_kwargs={"tokenizer": tokenizer}
)

print(f"Tokenized evaluation dataset size: {len(tokenized_eval_dataset)}")

# Compute metrics function
def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    # For causal LM, shift predictions and labels
    shift_predictions = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_predictions = shift_predictions.view(-1, shift_predictions.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss (cross entropy)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = loss_fct(shift_predictions, shift_labels)
    
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return {"eval_loss": loss.item(), "perplexity": perplexity.item()}

# Data collator
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# Setup evaluation
training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=4,
    dataloader_pin_memory=True,
    bf16=True,
    report_to=[],  # Disable wandb for evaluation
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run evaluation
print("🚀 Starting evaluation...")
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
