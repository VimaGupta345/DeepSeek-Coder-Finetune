import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import wandb

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

# -----------------------------
# Step 1: Dataset
# -----------------------------
def tokenize_and_format(example, tokenizer, max_length=512):
    code = example['text'].strip() + f"\n{EOT_TOKEN}"
    instruction = "Explain and rewrite the following OCaml code."
    full_text = instruction + "\n" + code
    inputs = tokenizer(full_text, max_length=max_length, truncation=True, padding="max_length")
    labels = inputs.input_ids.copy()
    ins_tokens = tokenizer(instruction, truncation=True, max_length=max_length).input_ids
    for i in range(len(ins_tokens)):
        if i < len(labels):
            labels[i] = IGNORE_INDEX
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels,
    }

# -----------------------------
# Step 2: Prepare
# -----------------------------
data_path = os.path.expanduser("~/scratch/finetune_data/ocaml_training/ocaml_instruct_format.jsonl")
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

print("Loading dataset with 80-10-10 split...")
ds = load_dataset("json", data_files=data_path, split="train")
train_val_test_split = ds.train_test_split(test_size=0.1, seed=42)
train_val_ds = train_val_test_split["train"]
test_ds = train_val_test_split["test"]
train_eval_split = train_val_ds.train_test_split(test_size=0.1111, seed=42)
train_ds = train_eval_split["train"]
eval_ds = train_eval_split["test"]

print("Tokenizing datasets...")
train_set = train_ds.map(lambda x: tokenize_and_format(x, tokenizer), remove_columns=["text"])
eval_set = eval_ds.map(lambda x: tokenize_and_format(x, tokenizer), remove_columns=["text"])
test_set = test_ds.map(lambda x: tokenize_and_format(x, tokenizer), remove_columns=["text"])

# -----------------------------
# Step 3: Train
# -----------------------------
wandb.init(project="ocaml-lora")

deep_args = TrainingArguments(
    output_dir="output-ocaml-2e05-16",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    gradient_checkpointing=False,
    logging_dir="./wandb_logs",
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    max_steps=1500,
    num_train_epochs=3,
    report_to=["wandb"],
    run_name="deepseek-lora-ocaml-lr2e05-b16",
)

trainer = Trainer(
    model=model,
    args=deep_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
)

print("Training...")
model.train()
trainer.train()

print("Evaluating LoRA model...")
model.eval()
eval_ft = trainer.evaluate()
print("Finetuned eval_loss:", eval_ft['eval_loss'])

print("Saving adapter...")
model.save_pretrained("lora-ocaml-out/adapter")

# -----------------------------
# Step 4: Load models & Compare
# -----------------------------
print("Reloading base model (no LoRA)...")
base_model_clean = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)

base_args = TrainingArguments(
    output_dir="eval-base-out",
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
    fp16=True,
    report_to=[],
)

trainer_base = Trainer(
    model=base_model_clean,
    args=base_args,
    eval_dataset=test_set,
    tokenizer=tokenizer,
)

print("Evaluating clean base model on test set:")
eval_clean = trainer_base.evaluate()
print("Base eval_loss (test):", eval_clean["eval_loss"])

print("Reloading base model + LoRA adapter...")
base_model_with_lora = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model_with_adapter = PeftModel.from_pretrained(base_model_with_lora, "lora-ocaml-out/adapter")

trainer_lora = Trainer(
    model=model_with_adapter,
    args=base_args,
    eval_dataset=test_set,
    tokenizer=tokenizer,
)

print("Evaluating finetuned LoRA model on test set:")
eval_lora = trainer_lora.evaluate()
print("LoRA eval_loss (test):", eval_lora["eval_loss"])
