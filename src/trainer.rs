use candle_core::{Tensor, Result};
use candle_core::backprop::GradStore;
use candle_nn::{Optimizer, AdamW};
// use candle_nn::AdamW; // Check if this exists

pub struct UnslothTrainingArguments {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub steps: usize,
    pub warmup_steps: usize,
    pub gradient_accumulation_steps: usize,
}

pub struct UnslothTrainer {
    pub model: crate::models::llama::FastLlamaModel,
    pub args: UnslothTrainingArguments,
    pub optimizer: AdamW,
    pub accumulated_grads: Option<GradStore>,
}

impl UnslothTrainer {
    pub fn new(
        model: crate::models::llama::FastLlamaModel,
        args: UnslothTrainingArguments,
    ) -> Result<Self> {
        let vars = model.get_trainable_parameters();
        let params = candle_nn::ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        };
        let optimizer = AdamW::new(vars, params)?;
        
        Ok(Self {
            model,
            args,
            optimizer,
            accumulated_grads: None,
        })
    }

    pub fn train(&mut self, data_loader: &crate::dataprep::loader::TensorDataLoader) -> Result<()> {
        println!("Starting training...");
        let steps = self.args.steps; 
        let mut current_step = 0;
        
        loop {
            // iterator consumes references, so we call iter() each epoch or just once loop?
            // TensorDataLoader::iter() creates a new iterator.
            let iter = data_loader.iter();
            
            for batch_result in iter {
                if current_step >= steps {
                    println!("Reached max steps: {}", steps);
                    return Ok(());
                }
                
                let (batch, labels) = batch_result.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let loss = self.train_step(&batch, &labels, current_step)?;
                
                if current_step % 10 == 0 {
                    println!("Step {}: Loss = {:.4}", current_step, loss);
                }
                
                current_step += 1;
            }
            
            // If dataset exhausted and steps not reached, repeat (epochs)
            // For now, infinite loop until steps reached.
        }
    }

    pub fn train_step(&mut self, batch: &Tensor, labels: &Tensor, step_idx: usize) -> Result<f64> {
        // Forward pass
        let logits = self.model.forward(batch, 0)?;
        
        // Loss
        let loss = cross_entropy_loss(&logits, labels)?;
        
        // Gradient Accumulation
        let accum_steps = self.args.gradient_accumulation_steps;
        let scale = 1.0 / (accum_steps as f64);
        let scaled_loss = (loss.clone() * scale)?;
        
        // Backward
        let grads = scaled_loss.backward()?;
        
        if let Some(ref mut acc_grads) = self.accumulated_grads {
            // Merge grads into acc_grads
            // Iterate over all trainable parameters
            let params = self.model.get_trainable_parameters();
            for var in params {
                if let Some(new_g) = grads.get(&var) {
                    if let Some(old_g) = acc_grads.get(&var) {
                        let sum = (old_g + new_g)?;
                        acc_grads.insert(&var, sum);
                    } else {
                        acc_grads.insert(&var, new_g.clone());
                    }
                }
            }
        } else {
            self.accumulated_grads = Some(grads);
        }
        
        if (step_idx + 1) % accum_steps == 0 {
             if let Some(ref acc_grads) = self.accumulated_grads {
                 self.optimizer.step(acc_grads)?;
             }
             self.accumulated_grads = None;
        }
        
        Ok(loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?)
    }

    pub fn evaluate(&self, data_loader: &crate::dataprep::loader::TensorDataLoader) -> Result<f64> {
        println!("Starting evaluation...");
        let mut total_loss = 0.0;
        let mut steps = 0;
        
        for batch_result in data_loader.iter() {
            let (batch, labels) = batch_result.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            // Forward (start_pos 0, no cache)
            let logits = self.model.forward(&batch, 0)?;
            let loss = cross_entropy_loss(&logits, &labels)?;
            total_loss += loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
            steps += 1;
        }
        
        let avg_loss = if steps > 0 { total_loss / steps as f64 } else { 0.0 };
        println!("Evaluation Loss: {:.4}", avg_loss);
        Ok(avg_loss)
    }
}

fn cross_entropy_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    // logits: [b, seq, vocab]
    // labels: [b, seq]
    let (_b, _s, _v) = logits.dims3()?;
    let logits_flat = logits.flatten(0, 1)?; // [b*s, v]
    let labels_flat = labels.flatten(0, 1)?; // [b*s]
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
    // Gather log prob of true label
    let nll = log_probs.gather(&labels_flat.unsqueeze(1)?, candle_core::D::Minus1)?.squeeze(1)?; // [b*s]
    nll.flatten_all()?.mean(0)?.neg()
}
