from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, upload_folder

# Step 1: Load base model and LoRA adapter
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "llama-cbt-checkpoints/checkpoint-250"  # Your LoRA checkpoint directory
merged_output_dir = "merged_llama3_lora"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

print("Loading LoRA adapter...")
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

# Step 2: Merge and save
print("Merging adapter into base model...")
merged_model = adapter_model.merge_and_unload()
merged_model.save_pretrained(merged_output_dir)

# Save tokenizer
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_output_dir)

# Step 3: Upload to Hugging Face Hub
repo_name = "CBT-Copilot"
repo_id = "thillaic/CBT-Copilot"  # Replace with your HF username

print(f"Creating HF repo: {repo_name}")
api = HfApi()
api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)


print("Uploading to Hugging Face Hub...")
upload_folder(folder_path=merged_output_dir, repo_id=repo_id, repo_type="model")

print("✅ Upload complete!")
