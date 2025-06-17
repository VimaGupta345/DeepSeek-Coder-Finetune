import os, gc, json, random, copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List

import torch
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments as HFTrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Argument classes                                                       │
# ╰──────────────────────────────────────────────────────────────────────────╯

@dataclass
class DataArguments:
    """Only needs local JSONL paths now."""

    data_path: str = field(metadata={"help": "JSON/JSONL file with a text field."})
    eval_mode: bool = field(
        default=False,
        metadata={"help": "Skip training, evaluate base + saved finetuned model."},
    )
    sample_limit: Optional[int] = field(
        default=None, metadata={"help": "Optionally cap the dataset to first N rows."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        metadata={"help": "Base model or already‑finetuned checkpoint."},
    )
    finetuned_model_path: str = field(
        default="finetuned_model",
        metadata={"help": "Where to load *and/or* save the LoRA‑merged model."},
    )

@dataclass
class TrainingArguments(HFTrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    logging_steps: int = field(default=50)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    max_steps: int = field(default=1500)
    num_train_epochs: int = field(default=3)
    run_name: Optional[str] = field(default="deepseek‑lora‑run")
    report_to: List[str] = field(default_factory=lambda: ["none"])  # set to wandb if you wish

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Helper functions                                                       │
# ╰──────────────────────────────────────────────────────────────────────────╯

def build_instruction_prompt() -> str:
    # Single generic instruction (fits code‑2 behaviour).
    return (
        """
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non‑computer science questions, you will refuse to answer.
### Instruction:
Explain and rewrite the following OCaml code:
### Response:
""".lstrip()
    )


def tokenize_examples(examples, tokenizer):
    """Tokenises according to code‑2 logic (prompt + target)."""
    prompt = build_instruction_prompt()
    sources = [prompt for _ in examples["text"]]
    targets = [f"{code}\n{EOT_TOKEN}" for code in examples["text"]]

    model_inputs = tokenizer(sources, add_special_tokens=False)
    target_inputs = tokenizer(targets, add_special_tokens=False)

    input_ids, labels = [], []
    for src_ids, tgt_ids in zip(model_inputs["input_ids"], target_inputs["input_ids"]):
        ids = src_ids + tgt_ids + [tokenizer.eos_token_id]
        labs = [-100] * len(src_ids) + tgt_ids + [tokenizer.eos_token_id]
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(labs, dtype=torch.long))
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ids  = [torch.tensor(x["input_ids"], dtype=torch.long) for x in instances]
        lbls = [torch.tensor(x["labels"],    dtype=torch.long) for x in instances]
        ids  = torch.nn.utils.rnn.pad_sequence(ids,  batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lbls = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": ids,
            "labels": lbls,
            "attention_mask": ids.ne(self.tokenizer.pad_token_id),
        }

# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Main                                                                   │
# ╰──────────────────────────────────────────────────────────────────────────╯

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files=data_args.data_path, split="train")
    if data_args.sample_limit:              # default 1 000
        ds = ds.select(range(data_args.sample_limit))

    # 80‑10‑10 split
    tvt = ds.train_test_split(test_size=0.1, seed=42)
    test_ds = tvt["test"]
    tv = tvt["train"].train_test_split(test_size=0.1111, seed=42)
    train_ds, eval_ds = tv["train"], tv["test"]

    # ─── Tokenise (assign back!) ───────────────────────────
    train_ds = train_ds.map(
        tokenize_examples, batched=True,
        remove_columns=train_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    eval_ds = eval_ds.map(
        tokenize_examples, batched=True,
        remove_columns=eval_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    test_ds = test_ds.map(
        tokenize_examples, batched=True,
        remove_columns=test_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    collator = DataCollatorForSupervisedDataset(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if data_args.eval_mode:
        print("\n[Eval‑only] …")
        print("Base:", trainer.evaluate(eval_dataset=test_ds))
        return

    trainer.train()
    print("Test:", trainer.evaluate(eval_dataset=test_ds))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(model_args.finetuned_model_path)
    tokenizer.save_pretrained(model_args.finetuned_model_path)


if __name__ == "__main__":
    main()