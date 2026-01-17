// Interface for RoPE embedding
// Corresponds to unsloth/kernels/rope_embedding.py

pub struct RopeEmbedding;

impl RopeEmbedding {
    /// CPU Reference implementation of RoPE
    /// Assumes Q and K are [batch, seq_len, n_heads, head_dim] flattened
    /// Cos and Sin are [max_seq_len, head_dim] flattened (pre-computed cos/sin)
    /// position_ids is [batch, seq_len] flattened
    pub fn apply(
        q: &mut [f32],
        k: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        position_ids: &[i64],
        batch_size: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        let n_tokens = batch_size * seq_len;
        let dim = head_dim;
        let half_dim = dim / 2;

        // Ensure we have enough data
        assert!(q.len() >= n_tokens * n_heads * dim);
        assert!(k.len() >= n_tokens * n_heads * dim); // K might have fewer heads (GQA), but for simple impl assuming equal
        assert!(position_ids.len() >= n_tokens);

        for i in 0..n_tokens {
            let pos = position_ids[i] as usize;
            
            // Get pointers to cos/sin for this position
            // Cos/Sin shape: [max_seq, dim]
            let cos_start = pos * dim;
            let sin_start = pos * dim;
            
            for h in 0..n_heads {
                let token_head_start = (i * n_heads + h) * dim;
                
                // Rotation logic:
                // x = [x1, x2]
                // out = [x1*cos - x2*sin, x2*cos + x1*sin]
                // where x1 is first half, x2 is second half
                
                for j in 0..half_dim {
                    let idx1 = token_head_start + j;
                    let idx2 = token_head_start + half_dim + j;
                    
                    let q1 = q[idx1];
                    let q2 = q[idx2];
                    
                    let k1 = k[idx1];
                    let k2 = k[idx2];
                    
                    // Cos/Sin values for this frequency component
                    let c = cos[cos_start + j];
                    let s = sin[sin_start + j];
                    
                    // Q rotation
                    q[idx1] = q1 * c - q2 * s;
                    q[idx2] = q2 * c + q1 * s;
                    
                    // K rotation
                    k[idx1] = k1 * c - k2 * s;
                    k[idx2] = k2 * c + k1 * s;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_rotation_basic() {
        let batch_size = 1;
        let seq_len = 1;
        let n_heads = 1;
        let head_dim = 2; // half_dim = 1

        let mut q = vec![1.0, 0.0];
        let mut k = vec![1.0, 0.0];
        // For head_dim=2, half_dim=1. j=0.
        // cos[0] used.
        let cos = vec![1.0, 1.0]; 
        let sin = vec![0.0, 0.0];
        let position_ids = vec![0];

        RopeEmbedding::apply(
            &mut q, &mut k, &cos, &sin, &position_ids,
            batch_size, seq_len, n_heads, head_dim
        );

        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_rotation_90_degrees() {
        // Rotate [1, 0] by 90 degrees
        // Cos=0, Sin=1
        // x_new = [1*0 - 0*1, 0*0 + 1*1] = [0, 1]
        
        let batch_size = 1;
        let seq_len = 1;
        let n_heads = 1;
        let head_dim = 2;

        let mut q = vec![1.0, 0.0];
        let mut k = vec![1.0, 0.0];
        
        let cos = vec![0.0, 0.0]; 
        let sin = vec![1.0, 1.0]; 
        
        let position_ids = vec![0];

        RopeEmbedding::apply(
            &mut q, &mut k, &cos, &sin, &position_ids,
            batch_size, seq_len, n_heads, head_dim
        );
        
        assert!((q[0] - 0.0).abs() < 1e-6, "q[0] was {}", q[0]);
        assert!((q[1] - 1.0).abs() < 1e-6, "q[1] was {}", q[1]);
    }
}
