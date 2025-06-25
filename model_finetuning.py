# install: pip install transformers accelerate peft trl datasets bitsandbytes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

# Load dataset
dataset = load_dataset("json", data_files="train.jsonl")["train"]

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Formatting function
def formatting_func(example):
    return f"{example['prompt']}{example['response']}"

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
args = TrainingArguments(
    output_dir="llama-cbt-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=100,
    save_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
)

# Trainer
# Apply formatting manually to dataset
def formatting_func(example):
    return {"text": f"{example['prompt']}{example['response']}"}

dataset = dataset.map(formatting_func)
dataset = dataset.rename_columns({"response": "completion"})
# Reduce dataset size for faster debugging
dataset = dataset.select(range(1000))


# Now remove formatting_func from the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    peft_config=peft_config,
)

trainer.train()
