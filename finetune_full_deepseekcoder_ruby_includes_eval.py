#!/usr/bin/env python
import os
import json
import copy
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import torch
import torch.distributed
import transformers
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct"
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="data/ruby_instruct_format.jsonl",
        metadata={"help": "Output path for processed JSONL."},
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Validation JSONL file path."}
    )
    hf_dataset_name: str = field(
        default="Nan-Do/instructional_code-search-net-ruby",
        metadata={"help": "HuggingFace dataset name."},
    )
    hf_train_split: str = field(
        default="train[:48000]",
        metadata={"help": "Training split spec for HF dataset."},
    )
    hf_eval_split: str = field(
        default="train[48000:]",
        metadata={"help": "Evaluation split spec for HF dataset."},
    )
    preprocess_only: bool = field(
        default=False, metadata={"help": "If true, only preprocess and exit."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded/truncated."},
    )
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = field(default="deepseek-full-finetune-ruby")
    logging_dir: Optional[str] = field(default="./wandb_logs")
    logging_steps: int = field(
        default=50, metadata={"help": "Log every X steps to W&B"}
    )
    evaluation_strategy: str = field(default="steps")
    eval_strategy: str = field(default="steps")
    do_eval: bool = field(default=True)
    eval_steps: int = field(default=4)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=6)
    max_steps: int = field(default=1500, metadata={"help": "Max training steps"})
    num_train_epochs: int = field(default=3)
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Load best model at end (requires matching strategies)"},
    )
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    warmup_steps: int = field(default=100)
    lr_scheduler_type: str = field(default="cosine")
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)
    label_names: List[str] = field(default_factory=lambda: ["labels"])



def process_hf_dataset(data_args: DataArguments):
    """Download and process HuggingFace dataset into instruction format."""
    os.makedirs(os.path.dirname(data_args.data_path), exist_ok=True)
    
    # Process training split
    print(f"Downloading {data_args.hf_dataset_name} training split {data_args.hf_train_split}...")
    train_ds = load_dataset(data_args.hf_dataset_name, split=data_args.hf_train_split)
    
    # Process evaluation split
    print(f"Downloading {data_args.hf_dataset_name} evaluation split {data_args.hf_eval_split}...")
    eval_ds = load_dataset(data_args.hf_dataset_name, split=data_args.hf_eval_split)
    
    print(f"Writing {len(train_ds)} training examples to {data_args.data_path}...")
    train_count = 0
    with open(data_args.data_path, "w") as f:
        for ex in train_ds:
            instruction = ex.get("INSTRUCTION", "").strip()
            response = ex.get("RESPONSE", "").strip()
            if not instruction or not response:
                continue
                
            entry = {
                "instruction": instruction,
                "output": response,
            }
            f.write(json.dumps(entry) + "\n")
            train_count += 1
    
    # Write evaluation data
    eval_data_path = data_args.data_path.replace(".jsonl", "_eval.jsonl")
    print(f"Writing {len(eval_ds)} evaluation examples to {eval_data_path}...")
    eval_count = 0
    with open(eval_data_path, "w") as f:
        for ex in eval_ds:
            instruction = ex.get("INSTRUCTION", "").strip()
            response = ex.get("RESPONSE", "").strip()
            if not instruction or not response:
                continue
                
            entry = {
                "instruction": instruction,
                "output": response,
            }
            f.write(json.dumps(entry) + "\n")
            eval_count += 1
    
    print(f"✅ Saved {train_count} training examples and {eval_count} evaluation examples")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def build_instruction_prompt(instruction: str) -> str:
    return f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:
"""

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
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

def train_tokenize_function(examples, tokenizer):
    EOT_TOKEN = "<|EOT|>"
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def load_datasets_from_hf(data_args: DataArguments, training_args: TrainingArguments):
    """Load train and eval datasets directly from HuggingFace with specified splits."""
    
    # Load training dataset
    if training_args.local_rank > 0:
        torch.distributed.barrier()
    
    print(f"Loading training dataset: {data_args.hf_dataset_name} split {data_args.hf_train_split}")
    train_raw_dataset = load_dataset(
        data_args.hf_dataset_name,
        split=data_args.hf_train_split,
        cache_dir=training_args.cache_dir
    )
    
    print(f"Loading evaluation dataset: {data_args.hf_dataset_name} split {data_args.hf_eval_split}")
    eval_raw_dataset = load_dataset(
        data_args.hf_dataset_name,
        split=data_args.hf_eval_split,
        cache_dir=training_args.cache_dir
    )
    
    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    # Rename columns to match what the script expects and remove original columns.
    train_raw_dataset = train_raw_dataset.rename_column("INSTRUCTION", "instruction")
    train_raw_dataset = train_raw_dataset.rename_column("RESPONSE", "output")
    eval_raw_dataset = eval_raw_dataset.rename_column("INSTRUCTION", "instruction")
    eval_raw_dataset = eval_raw_dataset.rename_column("RESPONSE", "output")
    
    # Remove all other columns
    train_column_names = train_raw_dataset.column_names
    for col in train_column_names:
        if col not in ['instruction', 'output']:
            train_raw_dataset = train_raw_dataset.remove_columns(col)

    eval_column_names = eval_raw_dataset.column_names
    for col in eval_column_names:
        if col not in ['instruction', 'output']:
            eval_raw_dataset = eval_raw_dataset.remove_columns(col)
            
    return train_raw_dataset, eval_raw_dataset

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Preprocess only mode
    if data_args.preprocess_only:
        print("🔄 Preprocessing dataset only...")
        process_hf_dataset(data_args)
        print("✅ Preprocessing completed. Exiting.")
        return

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Print training arguments for debugging
    print("=" * 100)
    print(training_args)
    print("=" * 100)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"PAD Token: {tokenizer.pad_token} {tokenizer.pad_token_id}")
    print(f"BOS Token {tokenizer.bos_token} {tokenizer.bos_token_id}")
    print(f"EOS Token {tokenizer.eos_token} {tokenizer.eos_token_id}")
    print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))

    # Load datasets directly from HuggingFace
    train_raw_dataset, eval_raw_dataset = load_datasets_from_hf(data_args, training_args)

    # Tokenize datasets - dataset now has 'instruction' and 'output' fields
    train_dataset = train_raw_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=train_raw_dataset.column_names,
        load_from_cache_file=True,
        desc="Running Encoding on Train Set",
        fn_kwargs={"tokenizer": tokenizer}
    )

    eval_dataset = eval_raw_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=eval_raw_dataset.column_names,
        load_from_cache_file=True,
        desc="Running Encoding on Eval Set",
        fn_kwargs={"tokenizer": tokenizer}
    )

    if training_args.local_rank == 0:
        print(f"Training dataset samples: {len(train_dataset)}")
        print(f"Evaluation dataset samples: {len(eval_dataset)}")
        for index in range(3):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Decoded sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    # Data collator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    # Initialize callbacks
    callbacks = []

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    #Evaluate first
    #print("*** Evaluate ***")
    #trainer.evaluate()

    # Training
    print("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
