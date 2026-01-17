use candle_core::{Tensor, Result, Device, Var, DType};
use candle_core::quantized::QMatMul;
// use crate::kernels::fast_lora::matmul_lora;
use candle_nn::{Linear, Module};

#[derive(Clone)]
pub enum BaseLinear {
    Standard(Linear),
    Quantized(QMatMul),
}

impl BaseLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            BaseLinear::Standard(l) => l.forward(x),
            BaseLinear::Quantized(q) => q.forward(x),
        }
    }
}

#[derive(Clone)]
pub struct LoRALinear {
    pub old_weight: BaseLinear,
    pub a: Option<Var>,
    pub b: Option<Var>,
    pub scale: f64,
    pub r: usize,
    pub dropout: f64,
}

impl LoRALinear {
    pub fn new(
        old_weight: BaseLinear,
        in_features: usize,
        out_features: usize,
        r: usize,
        lora_alpha: f64,
        dropout: f64,
        device: &Device,
    ) -> Result<Self> {
        let a = if r > 0 {
            Some(Var::from_tensor(&Tensor::randn(0f32, 1.0, (r, in_features), device)?)?)
        } else {
            None
        };
        let b = if r > 0 {
            Some(Var::from_tensor(&Tensor::zeros((out_features, r), DType::F32, device)?)?)
        } else {
            None
        };

        let scale = if r > 0 { lora_alpha / (r as f64) } else { 0.0 };

        Ok(Self {
            old_weight,
            a,
            b,
            scale,
            r,
            dropout,
        })
    }

    pub fn inject_lora(&mut self, in_features: usize, out_features: usize, r: usize, lora_alpha: f64, dropout: f64, device: &Device) -> Result<()> {
        let a = Some(Var::from_tensor(&Tensor::randn(0f32, 1.0, (r, in_features), device)?)?);
        let b = Some(Var::from_tensor(&Tensor::zeros((out_features, r), DType::F32, device)?)?);
        
        self.a = a;
        self.b = b;
        self.r = r;
        self.scale = lora_alpha / (r as f64);
        self.dropout = dropout;
        
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward
        let output = self.old_weight.forward(x)?;

        if self.r == 0 {
             return Ok(output);
        }
        
        // LoRA forward
        let a = self.a.as_ref().unwrap();
        let b = self.b.as_ref().unwrap();

        // (x @ A.T @ B.T) * scale
        // x: [batch, seq, in]
        // a: [r, in] -> a.t: [in, r]
        // b: [out, r] -> b.t: [r, out]
        
        // We need to flatten x if 3D, or use broadcast logic
        // But for LoRA we implemented manual matmul in fast_lora.rs? 
        // No, we did standard broadcast matmul there or here.
        // Let's use standard matmul here for simplicity unless using fast_lora kernel.
        // wait, I previously used `kernels::fast_lora::apply_lora` but I replaced `LoRALinear` to be self-contained in `src/models/lora.rs`?
        // Let's check the previous `LoRALinear::forward` content.
        
        // Previous content used `matmul_lora`? No, I rewrote it to be manual in `lora.rs`?
        // Let's implement standard matmul here.
        
        let x_in = if x.rank() == 3 {
            x.flatten(0, 1)?
        } else {
            x.clone()
        };
        
        let a_t = a.as_tensor().t()?;
        let b_t = b.as_tensor().t()?;
        
        let xa = x_in.matmul(&a_t)?;
        let xab = xa.matmul(&b_t)?;
        let lora_out = (xab * self.scale)?;
        
        let lora_out = if x.rank() == 3 {
             lora_out.reshape(output.shape())?
        } else {
             lora_out
        };
        
        output + lora_out
    }

    pub fn get_trainable_parameters(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        if let Some(a) = &self.a { vars.push(a.clone()); }
        if let Some(b) = &self.b { vars.push(b.clone()); }
        vars
    }

    pub fn get_lora_tensors(&self, name_prefix: &str) -> std::collections::HashMap<String, Tensor> {
        let mut map = std::collections::HashMap::new();
        if let Some(a) = &self.a {
            map.insert(format!("{}.lora_A.weight", name_prefix), a.as_tensor().clone());
        }
        if let Some(b) = &self.b {
            map.insert(format!("{}.lora_B.weight", name_prefix), b.as_tensor().clone());
        }
        map
    }

    pub fn merge(&self) -> Result<Tensor> {
        if self.r == 0 {
             // Just return base weight as tensor
             match &self.old_weight {
                 BaseLinear::Standard(l) => Ok(l.weight().clone()),
                 BaseLinear::Quantized(q) => {
                     // Dequantize logic. 
                     // QMatMul typically doesn't hold the dequantized weight.
                     // We need to dequantize it.
                     // In candle 0.4+, QMatMul usually doesn't expose strict dequantize easily without internal methods.
                     // However, we can use q.dequantize()? if available.
                     // Let's assume standard behavior:
                     // Or iterate blocks? No.
                     // Let's rely on forward pass of identity? No VRAM limit.
                     // candle_core::quantized::QTensor has dequantize.
                     // QMatMul stores Arc<QTensor>.
                     // Not accessible directly via QMatMul usually as fields are private (or public?).
                     // If we can't easily dequantize, we might be stuck.
                     // BUT, unsloth_rs/src/models/weights.rs has `WeightLoader`.
                     // Maybe we should fail if we can't merge 4bit?
                     // Wait, candle-core quantized usually exposes `dequantize`.
                     // Let's try `q.dequantize(self.device)?`.
                     // If compilation fails, we find another way.
                     // Actually `qt.dequantize(&device)` is common.
                     // Let's try to access the inner qtensor if possible, or use a workaround.
                     // Workaround: `q.forward` on rows one by one? Slow.
                     
                     // Let's try `q.dequantize(device)` assuming it exists or `q.weight().dequantize(device)`.
                     panic!("Merging 4-bit weights not fully implemented without deep candle access. Use 16-bit loading for full export.");
                 }
             }
        } else {
             let base = match &self.old_weight {
                 BaseLinear::Standard(l) => l.weight().clone(),
                 BaseLinear::Quantized(_) => {
                     // Same issue.
                     panic!("Merging 4-bit weights not fully implemented.");
                 }
             };
             
             let a = self.a.as_ref().unwrap().as_tensor();
             let b = self.b.as_ref().unwrap().as_tensor();
             let scale = self.scale;
             
             // W_new = W_base + B @ A * scale
             // B: [out, r], A: [r, in] -> B @ A: [out, in]
             // Standard LoRA: weight += (B @ A) * scale
             // Check shapes:
             // Linear weight is [out, in].
             // a: [r, in], b: [out, r].
             // b.matmul(&a)? -> [out, in]. Correct.
             
             let delta = b.matmul(a)?;
             let delta = (delta * scale)?;
             
             Ok((base + delta)?)
        }
    }

    pub fn merge_weights(&mut self) -> Result<()> {
        if self.r == 0 {
            return Ok(());
        }
        
        let merged_weight = self.merge()?;
        // Create new standard linear
        let new_linear = Linear::new(merged_weight, None); // TODO: Handle bias if original had it? LoRA usually applied to weight.
        
        self.old_weight = BaseLinear::Standard(new_linear);
        self.a = None;
        self.b = None;
        self.r = 0;
        self.scale = 0.0;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{Linear, linear, VarBuilder};

    #[test]
    // #[ignore] // FIXME: persistent shape mismatch error despite correct logic
    fn test_lora_linear_forward() -> Result<()> {
        let device = Device::Cpu;
        
        // Dimensions
        let in_dim = 10;
        let out_dim = 5;
        let r = 2;
        let hidden_dim = 10; // seq length or batch size factor
        
        // Create Linear layer manually to avoid VarBuilder issues in test
        let w = Tensor::randn(0f32, 1.0, (out_dim, in_dim), &device)?;
        let w_linear = Linear::new(w.clone(), None);
        
        let lora = LoRALinear::new(w_linear, in_dim, out_dim, r, 1.0, 0.0, &device)?;
        
        // Input: [1, 2, in]
        let x = Tensor::randn(0f32, 1.0, (1, 2, in_dim), &device)?;
        
        let output = lora.forward(&x)?;
        
        assert_eq!(output.dims(), &[1, 2, out_dim]);
        
        // Verify output roughly matches expectation logic manually implemented (optional)
        // Since B is init to zeros, output initially should be exactly X @ W^T.
        // Needs flattening for 3D input
        let x_flat = x.flatten(0, 1)?;
        let expected_flat = x_flat.matmul(&w.t()?)?;
        let expected = expected_flat.reshape((1, 2, out_dim))?;
        
        // Check differences
        let diff = (output - expected)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        
        println!("Diff with clean forward: {}", diff);
        assert!(diff < 1e-5, "Output should match base layer initially (B=0)");
        
        Ok(())
    }
}
