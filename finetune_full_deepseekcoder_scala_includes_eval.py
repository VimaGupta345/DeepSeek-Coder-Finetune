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
from datasets import load_dataset, Dataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct"
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="data/scala_instruct_format.jsonl",
        metadata={"help": "Output path for processed JSONL."},
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Validation JSONL file path."}
    )
    hf_dataset_name: str = field(
        default="sherwin6180/scala_instruct_format",
        metadata={"help": "HuggingFace dataset name."},
    )
    hf_train_split: str = field(
        default="train[:200000]",
        metadata={"help": "Training split spec for HF dataset."},
    )
    hf_eval_split: str = field(
        default="train[200000:220000]",
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
    run_name: Optional[str] = field(default="deepseek-full-finetune-scala")
    logging_dir: Optional[str] = field(default="./wandb_logs")
    logging_steps: int = field(
        default=50, metadata={"help": "Log every X steps to W&B"}
    )
    evaluation_strategy: str = field(default="steps")
    do_eval: bool = field(default=True)
    eval_steps: int = field(default=200)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    max_steps: int = field(default=1500, metadata={"help": "Max training steps"})
    num_train_epochs: int = field(default=3)
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Load best model at end (requires matching strategies)"},
    )
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=3)
    learning_rate: float = field(default=2e-5)
    warmup_steps: int = field(default=100)
    lr_scheduler_type: str = field(default="cosine")
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)

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

def load_local_datasets(data_args: DataArguments):
    """Load datasets from local JSONL file."""
    
    print(f"Loading training dataset from {data_args.data_path}")
    with open(data_args.data_path, 'r') as f:
        train_raw_data = [json.loads(line) for line in f]
    
    # Create training dataset from loaded data
    train_dataset = Dataset.from_list(train_raw_data)
    
    # Use 90% for training and 10% for evaluation (or create a simple split)
    total_samples = len(train_dataset)
    eval_start = int(0.9 * total_samples)
    
    train_indices = list(range(eval_start))
    eval_indices = list(range(eval_start, total_samples))
    
    train_raw_dataset = train_dataset.select(train_indices)
    eval_raw_dataset = train_dataset.select(eval_indices)
    
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(train_raw_dataset)}")
    print(f"Evaluation samples: {len(eval_raw_dataset)}")
    
    return train_raw_dataset, eval_raw_dataset


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    train_raw_dataset, eval_raw_dataset = load_local_datasets(data_args)

    # Tokenize datasets - dataset already has 'instruction' and 'output' fields
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
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

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
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

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
