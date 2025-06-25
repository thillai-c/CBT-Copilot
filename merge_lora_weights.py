from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base and adapter
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
adapter = PeftModel.from_pretrained(base, "llama-cbt-checkpoints/checkpoint-250")

# Merge and save
merged = adapter.merge_and_unload()
merged.save_pretrained("merged_llama3_lora")
