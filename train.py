#!/usr/bin/env python3
"""
Minimal QLoRA fine-tuning script for cybersecurity LLM
Token-optimized, production-ready
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Config
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # or "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./cybersec-model"
TRAIN_FILE = "dataset_train.json"
EVAL_FILE = "dataset_eval.json"

# QLoRA config - 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# LoRA config - efficient parameter updates
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model & tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Format dataset
def format_instruction(sample):
    """Convert to prompt format"""
    if sample["input"]:
        text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    else:
        text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
    return {"text": text}

# Load & prepare data
with open(TRAIN_FILE) as f:
    train_data = json.load(f)
with open(EVAL_FILE) as f:
    eval_data = json.load(f)

train_dataset = Dataset.from_list(train_data).map(format_instruction)
eval_dataset = Dataset.from_list(eval_data).map(format_instruction)

# Tokenize
def tokenize(sample):
    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = train_dataset.map(tokenize, remove_columns=["text", "instruction", "input", "output"])
eval_dataset = eval_dataset.map(tokenize, remove_columns=["text", "instruction", "input", "output"])

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    warmup_steps=50,
    optim="paged_adamw_8bit",
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

print("Starting training...")
trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
