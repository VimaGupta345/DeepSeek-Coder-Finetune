#!/usr/bin/env python

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Checkpoint and data path
checkpoint_path = "/scratch/shared_dir/finetuned_models/output/full_training_rust_20250626_014215"
data_path = "data/rust_instruct_format.jsonl"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

# Load dataset and get the same split used during training
full_dataset = load_dataset('json', data_files=data_path, split='train')

# Create the same train/eval split as in training (90/10 with seed=42)
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = split_dataset["test"]  # This is the evaluation set (last 10%)

print(f"Evaluation dataset size: {len(eval_dataset)}")

# Preprocess the evaluation data
def preprocess_function(examples):
    sources = [f"You are an AI programming assistant. {inst}\n### Response:" for inst in examples['instruction']]
    targets = [f"{out}\n<EOT>" for out in examples['output']]
    model_inputs = tokenizer(sources, max_length=1024, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=1024, truncation=True, padding='max_length').input_ids
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels
    ]
    return model_inputs

tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define evaluation
training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=1)
trainer = Trainer(model=model, args=training_args, eval_dataset=tokenized_eval_dataset, tokenizer=tokenizer)

# Run evaluation
print("Starting evaluation...")
eval_results = trainer.evaluate()

print("Evaluation Results:")
print(eval_results)
