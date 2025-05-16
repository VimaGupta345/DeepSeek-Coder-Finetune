import json
import os
from datasets import load_dataset

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Load dataset: take 100k examples
print("Loading dataset...")
ds = load_dataset("ammarnasr/the-stack-rust-clean", split="train[:100000]")

# Output path
output_path = "data/rust_instruct_format.jsonl"
count = 0

print("Processing and writing...")
with open(output_path, "w") as f:
    for example in ds:
        content = example.get("content", "").strip()
        if not content:
            continue

        instruction = "Explain and rewrite the following Rust code:"
        output = content

        entry = {"instruction": instruction, "output": output}
        f.write(json.dumps(entry) + "\n")
        count += 1

print(f"✅ Saved {count} examples to {output_path}")
