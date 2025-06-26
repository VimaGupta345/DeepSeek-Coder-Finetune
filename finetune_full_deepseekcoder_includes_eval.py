#!/usr/bin/env python
import os
import json
import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List

import torch
import torch.distributed
import transformers
from transformers import Trainer
from transformers.integrations import WandbCallback
from datasets import load_dataset

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

@dataclass
class DataArguments:
    data_path: str = field(
        default="data/rust_instruct_format2.jsonl",
        metadata={"help": "Output path for processed JSONL."},
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Validation JSONL file path."}
    )
    hf_dataset_name: str = field(
        default="Maverfrick/Rust_dataset",
        metadata={"help": "HuggingFace dataset name."},
    )
    hf_train_split: str = field(
        default="train[:100000]",
        metadata={"help": "Training split spec for HF dataset."},
    )
    hf_eval_split: str = field(
        default="train[100000:110000]",
        metadata={"help": "Evaluation split spec for HF dataset."},
    )
    preprocess_only: bool = field(
        default=False, metadata={"help": "If true, only preprocess and exit."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct"
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
    run_name: Optional[str] = field(default="deepseek-full-finetune")
    logging_dir: Optional[str] = field(default="./wandb_logs")
    logging_steps: int = field(
        default=50, metadata={"help": "Log every X steps to W&B"}
    )
    evaluation_strategy: str = field(default="steps")
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
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
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

    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss = loss_fct(shift_predictions, shift_labels)

    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return {"eval_loss": loss.item(), "perplexity": perplexity.item()}

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
            instruction = ex.get("instruction", "").strip()
            content = ex.get("response", "").strip()
            if not instruction or not content:
                continue
            entry = {
                "instruction": instruction,
                "output": content,
            }
            f.write(json.dumps(entry) + "\n")
            train_count += 1
    
    # Write evaluation data
    eval_data_path = data_args.data_path.replace(".jsonl", "_eval.jsonl")
    print(f"Writing {len(eval_ds)} evaluation examples to {eval_data_path}...")
    eval_count = 0
    with open(eval_data_path, "w") as f:
        for ex in eval_ds:
            instruction = ex.get("instruction", "").strip()
            content = ex.get("response", "").strip()
            if not instruction or not content:
                continue
            entry = {
                "instruction": instruction,
                "output": content,
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

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    """Tokenize function for training data using original logic."""
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
    
    return train_raw_dataset, eval_raw_dataset

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)

    # Preprocess dataset if requested
    if data_args.preprocess_only:
        process_hf_dataset(data_args)
        return

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if training_args.local_rank == 0:
        print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
        print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
        print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Configure model for training
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))

    # Load datasets with range-based splits
    train_raw_dataset, eval_raw_dataset = load_datasets_from_hf(data_args, training_args)

    # Tokenize datasets
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
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Setup callbacks
    callbacks = []
    if "wandb" in training_args.report_to:
        callbacks.append(WandbCallback)

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

    # Train
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
