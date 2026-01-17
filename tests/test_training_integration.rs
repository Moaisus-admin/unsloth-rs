
#[cfg(test)]
mod tests {
    use unsloth_rs::models::llama::FastLlamaModel;
    use unsloth_rs::models::loader::LlamaConfig;
    use unsloth_rs::trainer::{UnslothTrainer, UnslothTrainingArguments};
    use unsloth_rs::dataprep::loader::TensorDataLoader;
    use candle_core::{Device, Tensor, DType};
    use serde_json::Value;
    use candle_nn::{VarBuilder, Module};
    use std::collections::HashMap;

    #[test]
    fn test_training_loop_integration() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
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
        let model = FastLlamaModel::load(vb, &config)?;
        
        // 2. Mock Data
        // 4 samples, length 5
        let data = vec![
            serde_json::json!({"input_ids": [1, 2, 3, 4, 5]}),
            serde_json::json!({"input_ids": [6, 7, 8, 9, 10]}),
            serde_json::json!({"input_ids": [11, 12, 13, 14, 15]}),
            serde_json::json!({"input_ids": [16, 17, 18, 19, 20]}),
        ];
        
        let loader = TensorDataLoader::new(data, 2, device.clone(), false);
        
        // 3. Trainer
        let args = UnslothTrainingArguments {
            learning_rate: 1e-3,
            batch_size: 2,
            steps: 4, // 2 epochs
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
        };
        
        let mut trainer = UnslothTrainer::new(model, args)?;
        
        // 4. Train
        // Capture stdout to verify? Or just ensure no panic.
        trainer.train(&loader)?;
        
        // Evaluate on same loader for check
        let eval_loss = trainer.evaluate(&loader)?;
        assert!(eval_loss < 20.0); // Basic sanity check (random init is high)
        
        Ok(())
    }
}
