from unsloth_native import FastLanguageModel, UnslothTrainer
from transformers import TrainingArguments

# 1. Load Model (Compatible API)
# Uses local dummy model for demonstration
print("\n--- Using 'tiny_llama_real' for demo ---\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "tiny_llama_real",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
)

# 3. Train
# Create dummy dataset text
dataset = [
    {"text": "Refactoring code is like pruning a bonsai tree."}
]

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = TrainingArguments(
        output_dir = "outputs",
        learning_rate = 2e-4,
        max_steps = 1,
        per_device_train_batch_size = 1,
    )
)

trainer.train()

print("\nSuccess! Unsloth Native API works!")
