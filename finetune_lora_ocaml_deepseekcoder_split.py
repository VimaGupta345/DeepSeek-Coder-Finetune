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
from transformers import Trainer as HfTrainer
from transformers.integrations import WandbCallback
from datasets import load_dataset

# PEFT imports for LoRA
from peft import LoraConfig, get_peft_model

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
        default="~/scratch/finetune_data/ocaml_training/ocaml_instruct_format.jsonl",
        metadata={"help": "Path to the OCaml training data in JSONL format."},
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Validation JSONL file path."}
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
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded/truncated."},
    )
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = field(default="deepseek-lora-ocaml-b32-lr3e05")
    logging_dir: Optional[str] = field(default="./wandb_logs")
    logging_steps: int = field(
        default=25, metadata={"help": "Log every X steps to W&B"}
    )
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    per_device_eval_batch_size: int = field(default=2)
    eval_accumulation_steps: int = field(default=5)
    prediction_loss_only: bool = field(default=True)
    save_total_limit: int = field(default=3)
    max_steps: int = field(default=1500, metadata={"help": "Max training steps (~4h run)"})
    num_train_epochs: int = field(default=3)
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Load best model at end (requires matching strategies)"},
    )

class Trainer(HfTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raw_model = getattr(model, "module", model)
        outputs = raw_model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # shift for causal lm
    shift_logits = preds[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return {"eval_loss": loss.item(), "perplexity": perplexity.item()}

def train_tokenize_function(examples, tokenizer):
    sources = [build_instruction_prompt("Explain and rewrite the following OCaml code:")
               for _ in examples["text"]]
    targets = [f"{code}\n{EOT_TOKEN}" for code in examples["text"]]

    model_inputs = tokenizer(sources, add_special_tokens=False)
    target_inputs = tokenizer(targets, add_special_tokens=False)

    input_ids = []
    labels = []

    for src_ids, tgt_ids in zip(model_inputs["input_ids"], target_inputs["input_ids"]):
        ids   = src_ids + tgt_ids + [tokenizer.eos_token_id]
        labs  = [-100] * len(src_ids) + tgt_ids + [tokenizer.eos_token_id]

        input_ids.append(torch.tensor(ids,  dtype=torch.long))
        labels.append   (torch.tensor(labs, dtype=torch.long))

    return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # turn each list into a Tensor
        input_ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in instances]
        labels    = [torch.tensor(x['labels'],    dtype=torch.long) for x in instances]

        # now padding works
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels    = torch.nn.utils.rnn.pad_sequence(
            labels,    batch_first=True, padding_value=IGNORE_INDEX
        )
        return {
            'input_ids':      input_ids,
            'labels':         labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }

def safe_save_model_for_hf_trainer(trainer: HfTrainer, output_dir: str):
    """Collects model state dict to CPU and saves to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Tokenizer + base model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_disable()

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Load & split (from local JSONL)
    ds = load_dataset("json", data_files=data_args.data_path, split="train").select(range(1000))
    
    # 80-10-10 split for train, eval, and test
    train_val_test_split = ds.train_test_split(test_size=0.1, seed=42)
    train_val_ds = train_val_test_split["train"]
    test_ds = train_val_test_split["test"]

    train_eval_split = train_val_ds.train_test_split(test_size=0.11, seed=42)  # 0.11 * 0.9 == 0.1
    train_ds = train_eval_split['train']
    eval_ds = train_eval_split['test']

    print(f"Train examples: {len(train_ds)}   Val examples: {len(eval_ds)}   Test examples: {len(test_ds)}")

    # Tokenize
    train_ds = train_ds.map(
        train_tokenize_function,
        batched=True,
        remove_columns=train_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    eval_ds = eval_ds.map(
        train_tokenize_function,
        batched=True,
        remove_columns=eval_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    test_ds = test_ds.map(
        train_tokenize_function,
        batched=True,
        remove_columns=test_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # Collator & Trainer
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    callbacks = [WandbCallback] if "wandb" in training_args.report_to else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # if trainer.is_world_process_zero:
        # Evaluate on test set before training
        # print("--- Evaluating base model on test set ---")
        # pre_train_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="pre_train_test")
        # print(pre_train_metrics)

    # Train & save
    trainer.train()

    if trainer.is_world_process_zero:
        # Evaluate on test set after training
        print("\n--- Evaluating fine-tuned model on test set ---")
        post_train_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="post_train_test")
        print(post_train_metrics)


    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
