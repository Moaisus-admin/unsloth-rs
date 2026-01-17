use candle_core::{Result, Device, Tensor, DType};
use candle_nn::{VarBuilder, Linear};
use candle_core::quantized::{gguf_file, QMatMul};
use crate::models::lora::BaseLinear;

#[derive(Clone)]
pub enum WeightLoader<'a> {
    Safetensors(VarBuilder<'a>),
    Gguf {
        content: &'a gguf_file::Content,
        path: std::path::PathBuf,
        prefix: String,
        device: Device,
    },
}

impl<'a> WeightLoader<'a> {
    pub fn pp(&self, s: &str) -> Self {
        match self {
            Self::Safetensors(vb) => Self::Safetensors(vb.pp(s)),
            Self::Gguf { content, path, prefix, device } => {
                let new_prefix = if prefix.is_empty() {
                    s.to_string()
                } else {
                    format!("{}.{}", prefix, s)
                };
                Self::Gguf {
                    content,
                    path: path.clone(),
                    prefix: new_prefix,
                    device: device.clone(),
                }
            }
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Safetensors(vb) => vb.device(),
            Self::Gguf { device, .. } => device,
        }
    }
    
    pub fn dtype(&self) -> DType {
         match self {
            Self::Safetensors(vb) => vb.dtype(),
            Self::Gguf { .. } => DType::F32, // GGUF dequants to F32 usually
        }
    }

    pub fn load_linear(&self, name: &str, in_dim: usize, out_dim: usize) -> Result<BaseLinear> {
        match self {
            Self::Safetensors(vb) => {
                let weight = vb.get((out_dim, in_dim), name)?;
                // Check if bias exists? Unsloth Llama usually no bias.
                // We'll assume no bias for now as per previous logic.
                let linear = Linear::new(weight, None);
                Ok(BaseLinear::Standard(linear))
            }
            Self::Gguf { content, path, prefix, device } => {
                // Name mapping logic
                // HF prefix: model.layers.0.self_attn.q_proj
                // GGUF: blk.0.attn_q.weight
                
                // We need to construct the full key.
                // current prefix + name
                let full_path = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{}.{}", prefix, name)
                };
                
                // Naive mapping attempt (User might need to improve this)
                // If the user passes "model.layers.0.self_attn.q_proj", we try to find it.
                // GGUF keys are flattened.
                
                // Let's look up exact match first.
                if let Some(_tensor_info) = content.tensor_infos.get(&full_path) {
                     // Found exact match.
                     // Load QMatMul
                     let mut file = std::fs::File::open(path)?;
                     let qtensor = content.tensor(&mut file, full_path.as_str(), device)?;
                     // QTensor -> QMatMul ? QTensor is usually QT.
                     // We need to create QMatMul from QTensor?
                     // candle_core::quantized::QMatMul::from_qtensor(qtensor)?
                     
                     // Wait, candle API:
                     // QMatMul::from_qtensor is regular way.
                     let qmm = QMatMul::from_qtensor(qtensor)?;
                     return Ok(BaseLinear::Quantized(qmm));
                }
                
                // If not found, try mapping logic?
                // TODO: Implement mapping for Llama GGUF keys.
                // For now, return Error to see what keys exist if we fail.
                
                Err(candle_core::Error::Msg(format!("GGUF tensor not found: {}", full_path)))
            }
        }
    }
    
    // For embeddings, RMSNorm etc, we usually want F32/Tensor, not Quantized.
    pub fn load_tensor(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
         match self {
            Self::Safetensors(vb) => vb.get(shape, name),
            Self::Gguf { content, path, prefix, device } => {
                 let full_path = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{}.{}", prefix, name)
                };
                
                // Load and dequantize to F32
                 let mut file = std::fs::File::open(path)?;
                 let qtensor = content.tensor(&mut file, full_path.as_str(), device)?;
                 let t = qtensor.dequantize(device)?;
                 // Reshape?
                 if t.dims() != shape {
                     t.reshape(shape)
                 } else {
                     Ok(t)
                 }
            }
         }
    }
}
