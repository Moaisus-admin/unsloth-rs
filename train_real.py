import os
import json
import numpy as np
import unsloth_rs
from transformers import AutoTokenizer

# Mock real dataset (Alpaca style)
DATASET = [
    {"instruction": "Give three tips for staying healthy.", "input": "", "output": "1. Eat a balanced diet.\n2. Exercise regularily.\n3. Get enough sleep."},
    {"instruction": "Calculate the sum.", "input": "5 + 10", "output": "The sum is 15."},
    {"instruction": "Write a poem about rust.", "input": "", "output": "Rust is fast,\nSafe and vast,\nMemory checks,\nNo segfault wrecks."}
]

def create_model_compatible_with_tokenizer(model_dir, vocab_size=32000):
    os.makedirs(model_dir, exist_ok=True)
    from safetensors.numpy import save_file
    
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64, # Small for speed
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
        
    tensors = {}
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    
    print(f"Creating dummy weights for vocab {vocab_size}...")
    # Float32 for compatibility
    tensors["model.embed_tokens.weight"] = np.random.randn(vocab_size, hidden).astype(np.float32) * 0.02
    tensors["model.norm.weight"] = np.ones(hidden, dtype=np.float32)
    tensors["lm_head.weight"] = np.random.randn(vocab_size, hidden).astype(np.float32) * 0.02
    
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(inter, hidden).astype(np.float32) * 0.02
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(inter, hidden).astype(np.float32) * 0.02
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(hidden, inter).astype(np.float32) * 0.02
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)

    save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    print(f"Created model at {model_dir}")

def train_real():
    model_name = "tiny_llama_real"
    
    # 1. Setup Tokenizer (Real)
    # Use unsloth's tokenizer or a small one? 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' or 'gpt2'?
    # TinyLlama uses Llama tokenizer (vocab 32000).
    try:
        print("Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        print(f"Failed to load tokenizer (maybe network?): {e}")
        # Fallback to simple char level or random? 
        return

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # 2. Setup Model (Rust)
    create_model_compatible_with_tokenizer(model_name, vocab_size)
    
    print("Loading Rust Model...")
    model = unsloth_rs.PyFastLlamaModel.from_pretrained(model_name, 2048)
    
    # 3. Data Prep (Python Side)
    print("Processing Data...")
    input_ids_batch = []
    for item in DATASET:
        prompt = f"{item['instruction']}\n{item['input']}\n\n{item['output']}"
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids_batch.append(ids)
        print(f"Sample encoded length: {len(ids)}")

    # Add LoRA Adapters
    print("Adding LoRA Adapters...")
    model.add_lora_adapters(16, 16.0, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # 4. Train (Rust Side)
    print("Initializing Trainer...")
    trainer = unsloth_rs.PyUnslothTrainer(model, 2e-4, 3) # 3 steps
    
    print("Starting Training on Real Text...")
    # Unsloth Reader expects batch_size. We pass all items.
    # Trainer handles batching? 
    # Current `train` takes full list and `batch_size`.
    trainer.train(input_ids_batch, 2)
    
    print("Training Complete!")
    model.save_pretrained("trained_real_model")
    print("Saved trained adapters.")

if __name__ == "__main__":
    train_real()
