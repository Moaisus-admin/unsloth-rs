import os
import json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import unsloth_rs

def test_export():
    model_dir = "dummy_model"
    # Ensure dummy model exists (re-use creating logic for safety or assume existing)
    if not os.path.exists(model_dir):
        print("Model dir missing, recreating...")
        import test_binding
        test_binding.create_dummy_model(model_dir)

    print("Loading model...")
    model = unsloth_rs.PyFastLlamaModel.from_pretrained(model_dir, 128)
    
    # Initialize trainer to ensure LoRA is set up (though calling forward/train works)
    trainer = unsloth_rs.PyUnslothTrainer(model, 3e-4, 1)
    
    # Run one step to update weights (optional, but good for reality check)
    dataset = [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]]
    trainer.train(dataset, 1)
    
    print("Merging and saving...")
    out_path = "merged_model.safetensors"
    model.merge_and_save(out_path)
    
    if os.path.exists(out_path):
        print(f"Success: {out_path} created.")
        
        # Verify keys
        with safe_open(out_path, framework="np", device="cpu") as f:
            keys = f.keys()
            print(f"Keys found: {len(keys)}")
            # Check for a specific key
            if "model.layers.0.self_attn.q_proj.weight" in keys:
                print("Found merged q_proj weight.")
            else:
                print("Error: detailed keys missing.")
                print(keys)
                exit(1)
                
            # Check that lora keys are NOT present
            has_lora = any("lora_A" in k for k in keys)
            if has_lora:
                print("Error: LoRA keys found! Should be merged.")
                exit(1)
            else:
                print("Verified: No LoRA keys remaining.")

    else:
        print("Error: File not created.")
        exit(1)

if __name__ == "__main__":
    test_export()
