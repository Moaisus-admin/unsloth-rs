// RMS LayerNorm kernel implementation

pub struct RMSLayerNorm;

impl RMSLayerNorm {
    /// CPU Reference implementation of RMS LayerNorm
    /// x: [batch * seq_len, hidden_size] flattened
    /// weight: [hidden_size]
    /// epsilon: stability constant
    pub fn apply(
        x: &mut [f32],
        weight: &[f32],
        epsilon: f32,
        n_rows: usize,
        n_cols: usize,
    ) {
        assert!(x.len() == n_rows * n_cols);
        assert!(weight.len() == n_cols);

        for i in 0..n_rows {
            let row_start = i * n_cols;
            let row_end = row_start + n_cols;
            let row = &mut x[row_start..row_end];
            
            // Calculate sum of squares
            let mut sum_sq = 0.0;
            for val in row.iter() {
                sum_sq += val * val;
            }
            
            let row_var = sum_sq / n_cols as f32;
            let inv_var = 1.0 / (row_var + epsilon).sqrt();
            
            // Normalize and scale
            for j in 0..n_cols {
                row[j] = row[j] * inv_var * weight[j];
            }
        }
    }
}

// Keep the function interface similar if needed, or deprecate the string placeholder
pub fn fast_rms_layernorm(
    x: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    n_rows: usize,
    n_cols: usize
) {
    RMSLayerNorm::apply(x, weight, epsilon, n_rows, n_cols);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_layernorm_basic() {
        let n_rows = 1;
        let n_cols = 2;
        let eps = 1e-5;
        
        // Input: [1.0, 1.0]
        // Mean square: (1+1)/2 = 1.0. inv_var = 1/sqrt(1+eps) approx 1.0.
        // Normed: [1.0, 1.0]
        // Weight: [1.0, 1.0]
        // Output: [1.0, 1.0]
        
        let mut x = vec![1.0, 1.0];
        let weight = vec![1.0, 1.0];
        
        RMSLayerNorm::apply(&mut x, &weight, eps, n_rows, n_cols);
        
        assert!((x[0] - 1.0).abs() < 1e-3);
        assert!((x[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_rms_layernorm_values() {
        let n_rows = 1;
        let n_cols = 2;
        let eps = 0.0; // Simplify math
        
        // Input: [2.0, 0.0]
        // Sum sq = 4.0. Mean sq = 2.0.
        // inv_var = 1/sqrt(2) = 0.7071
        // Normed: [2 * 0.7071, 0] = [1.4142, 0]
        // Weight: [2.0, 0.5]
        // Output: [2.8284, 0]
        
        let mut x = vec![2.0, 0.0];
        let weight = vec![2.0, 0.5];
        
        RMSLayerNorm::apply(&mut x, &weight, eps, n_rows, n_cols);
        
        assert!((x[0] - 2.0 * (2.0f32).sqrt()).abs() < 1e-4, "x[0] was {}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-4);
    }
}
