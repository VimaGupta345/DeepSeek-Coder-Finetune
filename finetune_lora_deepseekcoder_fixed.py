#!/usr/bin/env python
import os
import json
import torch
import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List

import math
import numpy as np
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
        default="data/rust_instruct_format.jsonl",
        metadata={"help": "Output path for processed JSONL."},
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Validation JSONL file path."}
    )
    hf_dataset_name: str = field(
        default="ammarnasr/the-stack-rust-clean",
        metadata={"help": "HuggingFace dataset name."},
    )
    hf_dataset_split: str = field(
        default="train[:100000]",
        metadata={"help": "Split spec for HF dataset."},
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

    # Learning rate - CRITICAL for LoRA
    learning_rate: float = field(
        default=1e-4,   # FullFT typically uses ~1-2e-5, LoRA requires 10x
        metadata={"help": "Learning rate (10x FullFT for LoRA per article)"}
    )

    # Batch sizes
    per_device_train_batch_size: int = field(default=32) # CRITICAL: global batch size should be kept ≤ 32
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)

    # Sequence length
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length."},
    )

    # Training duration - use ONE of these
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=-1)  # -1 = use epochs instead

    # Precision
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)

    # LR scheduler
    lr_scheduler_type: str = field(default="constant")
    warmup_ratio: float = field(default=0.00)

    # Logging
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = field(default="deepseek-lora-optimized")
    logging_dir: Optional[str] = field(default="./wandb_logs")
    logging_steps: int = field(default=50)
    logging_first_step: bool = field(default=True)

    # Evaluation
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)

    # Saving
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)

class Trainer(HfTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raw_model = getattr(model, "module", model)
        outputs = raw_model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    # HF sometimes returns a tuple/list; take logits
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]

    # Convert NumPy -> Torch
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Dtypes for CE
    predictions = predictions.float()
    labels = labels.long()

    # If this is generate() output (predict_with_generate=True), skip CE
    if predictions.ndim != 3:
        return {"eval_loss": float("nan"), "perplexity": float("inf")}

    # Causal shift
    shift_logits = predictions[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Flatten
    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    shift_labels = shift_labels.reshape(-1)

    # Mask (either do this OR use ignore_index in CE; doing both is fine)
    mask = shift_labels != IGNORE_INDEX
    if mask.sum().item() == 0:
        return {"eval_loss": float("nan"), "perplexity": float("inf")}

    shift_logits = shift_logits[mask]
    shift_labels = shift_labels[mask]

    # CE on filtered tokens
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)

    # Stable perplexity
    l = loss.item()
    ppl = math.exp(l) if l < 20 else float("inf")

    return {"eval_loss": l, "perplexity": ppl}


    # preds, labels = eval_pred
    # # shift for causal lm
    # shift_logits = preds[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    # loss = loss_fct(
    #     shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    # )
    # try:
    #     perplexity = torch.exp(loss)
    # except OverflowError:
    #     perplexity = float("inf")
    # return {"eval_loss": loss.item(), "perplexity": perplexity.item()}

def process_hf_dataset(data_args: DataArguments):
    os.makedirs(os.path.dirname(data_args.data_path), exist_ok=True)
    print(f"Downloading {data_args.hf_dataset_name} split {data_args.hf_dataset_split}...")
    ds = load_dataset(data_args.hf_dataset_name, split=data_args.hf_dataset_split)
    print(f"Writing {len(ds)} examples to {data_args.data_path}...")
    count = 0
    with open(data_args.data_path, "w") as f:
        for ex in ds:
            content = ex.get("content", "").strip()
            if not content:
                continue
            entry = {
                "instruction": "Explain and rewrite the following Rust code:",
                "output": content,
            }
            f.write(json.dumps(entry) + "\n")
            count += 1
    print(f"✅ Saved {count} examples")

def train_tokenize_function(examples, tokenizer):
    sources = [build_instruction_prompt(ins) for ins in examples["instruction"]]
    # Use tokenizer.eos_token (registered) instead of a raw EOT string
    targets = [f"{out}{tokenizer.eos_token}" for out in examples["output"]]

    # Tokenize sources and targets separately to get exact source lengths
    src_tok = tokenizer(sources, add_special_tokens=False)
    tgt_tok = tokenizer(targets, add_special_tokens=False)

    input_ids, labels = [], []
    max_len = tokenizer.model_max_length
    for s_ids, t_ids in zip(src_tok["input_ids"], tgt_tok["input_ids"]):
        ids = (s_ids + t_ids)[:max_len]
        lab = ids.copy()
        # mask the source portion only
        src_len = min(len(s_ids), max_len)
        lab[:src_len] = [IGNORE_INDEX] * src_len
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))
    return {"input_ids": input_ids, "labels": labels}

    # sources = [build_instruction_prompt(ins) for ins in examples["instruction"]]
    # targets = [f"{out}\n{EOT_TOKEN}" for out in examples["output"]]
    # tokenized = [
    #     tokenizer(s + t, truncation=True, max_length=tokenizer.model_max_length)
    #     for s, t in zip(sources, targets)
    # ]
    # input_ids = [torch.tensor(x["input_ids"]) for x in tokenized]
    # labels = [x.clone() for x in input_ids]
    # for lbl, x in zip(labels, tokenized):
    #     if tokenizer.eos_token_id in x["input_ids"]:
    #         src_len = x["input_ids"].index(tokenizer.eos_token_id) + 1
    #     else:
    #         src_len = 0
    #     lbl[:src_len] = IGNORE_INDEX
    # return {"input_ids": input_ids, "labels": labels}

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

    # Preprocess iff asked
    if data_args.preprocess_only:
        process_hf_dataset(data_args)
        return

    # Tokenizer + base model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    # Address issue: LLaMa-like tokenizers often have `pad_token=None`
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    # ThinkingMachine suggestion - memory efficiency
    model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Hard-fail if no LoRA layers were injected
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        try:
            from peft.tuners.lora.layer import LoraLayer
        except Exception:
            LoraLayer = None
    if LoraLayer is not None:
        hit = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    else:
        hit = [n for n, m in model.named_modules() if any(hasattr(m, attr) for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"))]

    assert len(hit) > 0, "No LoRA layers were injected. Check target_modules names."

    # print the trainable parameters
    model.print_trainable_parameters() 

    # Load & split
    ds = load_dataset("json", data_files=data_args.data_path, split="train")
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]
    print(f"Train examples: {len(train_ds)}   Val examples: {len(eval_ds)}")

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

    # Collator & Trainer
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    # redundant
    # callbacks = [WandbCallback] if "wandb" in training_args.report_to else []
    callbacks = []

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

    # Train & save
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
