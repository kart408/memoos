"""
Merges the LoRA adapter into the base model, producing a single
standalone model ready for GGUF conversion.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./finetuned_adapter"
OUTPUT_PATH = "./merged_model"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print("Done.")