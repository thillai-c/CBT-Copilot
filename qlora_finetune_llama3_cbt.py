import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import torch

# Load and format dataset
df = pd.read_csv("cbt.csv")
df = df[["input", "output"]].dropna()

def format_prompt(example):
    return f"""### Instruction:
{example['input']}

### Response:
{example['output']}"""

df["text"] = df.apply(format_prompt, axis=1)

# Save formatted data to JSONL
df[["text"]].to_json("formatted_cbt.jsonl", lines=True, orient="records")

from datasets import load_dataset
dataset = load_dataset("json", data_files="formatted_cbt.jsonl")["train"]

# Load model and tokenizer
model_name = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# LoRA config
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training args
training_args = TrainingArguments(
    output_dir="./llama3-cbt-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=1024
)

trainer.train()

# Save model
trainer.model.save_pretrained("./llama3-cbt-lora")
tokenizer.save_pretrained("./llama3-cbt-lora")