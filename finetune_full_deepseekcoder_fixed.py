#!/usr/bin/env python
import os
import json
import copy
import random
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import torch
import torch.distributed
import transformers
# from transformers import Trainer, TrainingArguments, set_seed
from transformers import Trainer, set_seed
from datasets import load_dataset

# Set seed for reproducibility
set_seed(42)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct"
    )

@dataclass
class DataArguments:
    # Local File loading
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local training JSONL file. If provided, uses local file instead of HF."},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local eval JSONL file. If not provided, splits from train."}
    )
    
    # Huggingface dataset loading
    hf_dataset_name: str = field(
        default="ammarnasr/the-stack-rust-clean",
        metadata={"help": "HuggingFace dataset name."},
    )
    hf_train_split: str = field(
        default="train[:250000]",
        metadata={"help": "Training split spec for HF dataset."},
    )
    hf_eval_split: str = field(
        default="train[250000:275000]",
        metadata={"help": "Evaluation split spec for HF dataset."},
    )

    # Common params
    min_code_length: int = field(
        default=100,
        metadata={"help": "Minimum code length to include"}
    )
    eval_split_ratio: float = field(
        default=0.1,
        metadata={"help": "If eval_data_path not provided, split this ratio from train"}
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length."},
    )
    
    # Training duration
    num_train_epochs: int = field(default=1)  # Article used 1 epoch
    max_steps: int = field(default=-1)
    
    # Batch sizes - conservative for full fine-tuning
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)  # Effective batch = 32
    
    # Learning rate and optimizer for full fine-tuning
    learning_rate: float = field(default=1e-5)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    weight_decay: float = field(default=0.05)  # Better for code SFT
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)  # Better for code SFT
    max_grad_norm: float = field(default=1.0)
    
    # Memory optimization
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    eval_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Accumulate eval predictions to avoid memory spikes"}
    )
    
    # Logging and evaluation
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = field(default="deepseek-full-ft-rust")
    logging_dir: Optional[str] = field(default="./wandb_logs")
    logging_steps: int = field(default=50)
    logging_first_step: bool = field(default=True)

    do_eval: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=250)
    
    # Saving
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=250)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Handle tuple predictions
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    
    # Convert to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Ensure correct dtypes
    predictions = predictions.float()
    labels = labels.long()
    
    # Check dimensions
    if predictions.ndim != 3:
        return {"eval_loss": float("nan"), "perplexity": float("inf")}
    
    # Shift for causal LM
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Mask ignored tokens
    mask = shift_labels != -100
    if mask.sum().item() == 0:
        return {"eval_loss": float("nan"), "perplexity": float("inf")}
    
    # Compute loss on non-ignored tokens
    loss = torch.nn.functional.cross_entropy(
        shift_logits[mask], 
        shift_labels[mask]
    )
    
    # Stable perplexity computation
    l = loss.item()
    ppl = math.exp(l) if l < 20 else float("inf")
    
    return {"eval_loss": l, "perplexity": ppl}

def build_instruction_prompt(instruction: str) -> str:
    return f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:
"""

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> Dict:
    """Preprocess the data by tokenizing.
    
    Args:
        sources: List of instruction prompts
        targets: List of target outputs
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    """
    input_ids_list = []
    labels_list = []
    
    for source, target in zip(sources, targets):
        # Tokenize source and target separately to get exact token boundaries
        source_ids = tokenizer(source, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        
        # Concatenate and truncate to max length
        input_ids = (source_ids + target_ids)[:max_length]
        
        # Create labels: mask source portion with -100
        labels = copy.deepcopy(input_ids)
        source_len = min(len(source_ids), max_length)
        labels[:source_len] = [-100] * source_len
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    return dict(input_ids=input_ids_list, labels=labels_list)

def train_tokenize_function(examples, tokenizer, max_length):
    """Tokenize training examples."""
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    # Use actual EOS token instead of custom EOT
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    
    data_dict = preprocess(sources, targets, tokenizer, max_length)
    return data_dict

def load_datasets_from_local(data_args: DataArguments, training_args: TrainingArguments):
    """Load train and eval datasets from local JSONL files."""
    
    print(f"Loading training dataset from local file: {data_args.data_path}")
    train_dataset = load_dataset(
        "json", 
        data_files=data_args.data_path, 
        split="train",
        cache_dir=training_args.cache_dir
    )
    
    # Load or split eval dataset
    if data_args.eval_data_path:
        print(f"Loading evaluation dataset from local file: {data_args.eval_data_path}")
        eval_dataset = load_dataset(
            "json", 
            data_files=data_args.eval_data_path, 
            split="train",
            cache_dir=training_args.cache_dir
        )
    else:
        print(f"Splitting {data_args.eval_split_ratio*100}% of training data for evaluation")
        split_dataset = train_dataset.train_test_split(
            test_size=data_args.eval_split_ratio, 
            seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    
    # Filter short examples
    print(f"Filtering examples shorter than {data_args.min_code_length} characters...")
    train_dataset = train_dataset.filter(
        lambda x: len(x.get('output', '')) >= data_args.min_code_length,
        desc="Filtering short training examples"
    )
    
    eval_dataset = eval_dataset.filter(
        lambda x: len(x.get('output', '')) >= data_args.min_code_length,
        desc="Filtering short eval examples"
    )
    
    print(f"✅ Loaded {len(train_dataset)} training examples and {len(eval_dataset)} eval examples")
    
    return train_dataset, eval_dataset

def load_datasets_from_hf(data_args: DataArguments, training_args: TrainingArguments):
    """Load train and eval datasets from HuggingFace with proper DDP barriers."""
    
    # is_main = training_args.local_rank in (-1, 0)
    
    # # Non-main ranks wait for main rank to download
    # if not is_main:
    #     torch.distributed.barrier()
    def maybe_barrier():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    is_main = training_args.local_rank in (-1, 0)
    if not is_main:
        maybe_barrier()


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
    
    # Main rank signals completion
    if is_main:
        # torch.distributed.barrier()
        maybe_barrier()

    
    # Convert to instruction format
    def convert_to_instruction_format(examples):
        instructions = []
        outputs = []
        
        for content in examples['content']:
            content = content.strip()
            instructions.append("Explain and rewrite the following Rust code")
            outputs.append(content)
        
        return {
            'instruction': instructions,
            'output': outputs
        }
    
    # Convert datasets
    train_raw_dataset = train_raw_dataset.map(
        convert_to_instruction_format,
        batched=True,
        remove_columns=train_raw_dataset.column_names,
        desc="Converting training dataset to instruction format"
    )
    
    eval_raw_dataset = eval_raw_dataset.map(
        convert_to_instruction_format,
        batched=True,
        remove_columns=eval_raw_dataset.column_names,
        desc="Converting eval dataset to instruction format"
    )
    
    # Filter short examples AFTER conversion
    train_raw_dataset = train_raw_dataset.filter(
        lambda x: len(x['output']) >= data_args.min_code_length,
        desc="Filtering short training examples"
    )
    
    eval_raw_dataset = eval_raw_dataset.filter(
        lambda x: len(x['output']) >= data_args.min_code_length,
        desc="Filtering short eval examples"
    )
    
    return train_raw_dataset, eval_raw_dataset

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model state dict to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    print("=" * 100)
    print(training_args)
    print("=" * 100)

    # Load tokenizer with proper configuration
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True, 
        use_fast=True  # Use fast tokenizer for speed
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Important for causal LM!
    
    print(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"Tokenizer loaded from {model_args.model_name_or_path}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    
    # Disable cache for gradient checkpointing
    model.config.use_cache = False
    
    # Enable FlashAttention2 if available (for H100/H200)
    try:
        model.config.attn_implementation = "flash_attention_2"
        print("✅ FlashAttention2 enabled")
    except:
        print("⚠️ FlashAttention2 not available, using default attention")
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if training_args.local_rank == 0:
        print(f"Model loaded from {model_args.model_name_or_path}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load datasets

    if data_args.data_path:
        print("📁 Loading datasets from local files...")
        train_raw_dataset, eval_raw_dataset = load_datasets_from_local(data_args, training_args)
    elif data_args.hf_dataset_name:
        print("🤗 Loading datasets from HuggingFace...")
        train_raw_dataset, eval_raw_dataset = load_datasets_from_hf(data_args, training_args)
    else:
        raise ValueError("Must provide either --data_path (local) or --hf_dataset_name (HuggingFace)")

    # Tokenize datasets - pass max_length as parameter
    train_dataset = train_raw_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=train_raw_dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing training set",
        fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}
    )

    eval_dataset = eval_raw_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=eval_raw_dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing eval set",
        fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.model_max_length}
    )

    if training_args.local_rank == 0:
        print(f"Training dataset samples: {len(train_dataset)}")
        print(f"Evaluation dataset samples: {len(eval_dataset)}")
        
        # Show sample with special tokens visible
        for index in random.sample(range(len(train_dataset)), min(2, len(train_dataset))):
            print(f"\n{'='*80}")
            print(f"Sample {index}:")
            print(f"Input IDs length: {len(train_dataset[index]['input_ids'])}")
            print(f"Labels length: {len(train_dataset[index]['labels'])}")
            print(f"Non-masked tokens: {sum(1 for x in train_dataset[index]['labels'] if x != -100)}")
            print(f"\nDecoded (first 200 tokens):")
            print(tokenizer.decode(train_dataset[index]['input_ids'][:200], skip_special_tokens=False))
            print(f"{'='*80}")

    # Data collator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )

    # Create trainer - use tokenizer parameter for better compatibility
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # Use tokenizer instead of processing_class for compatibility
        compute_metrics=compute_metrics,
    )

    # Training
    print("\n" + "="*100)
    print("*** Starting Full Fine-Tuning ***")
    print("="*100)
    train_result = trainer.train()
    
    # Save
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    print("\n" + "="*100)
    print("*** Training Complete ***")
    print("="*100)

if __name__ == "__main__":
    train()
