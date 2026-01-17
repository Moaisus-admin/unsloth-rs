
#[cfg(test)]
mod tests {
    use unsloth_rs::models::llama::FastLlamaModel;
    use unsloth_rs::models::loader::LlamaConfig;
    use candle_core::{Device, Tensor, DType, Var};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;
    use std::path::Path;
    use std::fs;

    #[test]
    fn test_save_lora_adapters() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let save_dir = "test_artifacts/save_test";
        if Path::new(save_dir).exists() {
            fs::remove_dir_all(save_dir).unwrap();
        }

        // 1. Mock Model
        let config = LlamaConfig {
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            vocab_size: 100,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 128,
            sliding_window: None,
            rope_theta: 10000.0,
        };
        
        // Initialize weights
        let mut map = HashMap::new();
        let mut add = |name: &str, shape: Vec<usize>| {
            let t = Tensor::randn(0.0f32, 1.0, shape.as_slice(), &device).unwrap();
            map.insert(name.to_string(), t);
        };
        add("model.embed_tokens.weight", vec![100, 16]);
        add("model.layers.0.self_attn.q_proj.weight", vec![16, 16]);
        add("model.layers.0.self_attn.k_proj.weight", vec![16, 16]);
        add("model.layers.0.self_attn.v_proj.weight", vec![16, 16]);
        add("model.layers.0.self_attn.o_proj.weight", vec![16, 16]);
        add("model.layers.0.mlp.gate_proj.weight", vec![64, 16]);
        add("model.layers.0.mlp.up_proj.weight", vec![64, 16]);
        add("model.layers.0.mlp.down_proj.weight", vec![16, 64]);
        add("model.layers.0.input_layernorm.weight", vec![16]);
        add("model.layers.0.post_attention_layernorm.weight", vec![16]);
        add("model.norm.weight", vec![16]);
        add("lm_head.weight", vec![100, 16]);
        
        let vb = VarBuilder::from_tensors(map, DType::F32, &device);
        let mut model = FastLlamaModel::load(vb, &config)?;
        
        // 2. Inject LoRA Adapters manually (Simulate training)
        let r = 4;
        let a_tensor = Tensor::randn(0.0f32, 1.0, (r, 16), &device)?;
        let b_tensor = Tensor::zeros((16, r), DType::F32, &device)?;
        
        let a_var = Var::from_tensor(&a_tensor)?;
        let b_var = Var::from_tensor(&b_tensor)?;
        
        // Inject into q_proj
        model.layers[0].self_attn.q_proj.a = Some(a_var);
        model.layers[0].self_attn.q_proj.b = Some(b_var);
        
        // 3. Save
        model.save_pretrained(save_dir).map_err(|e| candle_core::Error::Msg(e))?;
        
        // 4. Verify
        let save_path = Path::new(save_dir).join("adapter_model.safetensors");
        assert!(save_path.exists());
        
        let loaded = candle_core::safetensors::load(&save_path, &device)?;
        assert!(loaded.contains_key("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"));
        assert!(loaded.contains_key("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"));
        
        // Verify Content
        let loaded_a = loaded.get("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight").unwrap();
        // Compare with original
        let diff = (loaded_a - a_tensor)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5);
        
        // Clean up
        fs::remove_dir_all(save_dir).unwrap();
        
        Ok(())
    }
}
