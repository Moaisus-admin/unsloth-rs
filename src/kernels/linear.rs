use rayon::prelude::*;

/// CPU Reference Linear Layer: y = x * weight^T + bias
///
/// x: [batch_size, in_features]
/// weight: [out_features, in_features]
/// bias: Optional [out_features]
/// out: [batch_size, out_features]
pub fn linear_forward(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    _batch_size: usize,
    in_features: usize,
    out_features: usize,
) {
    // Simple parallel CPU matmul
    // Iterating over rows of x (batches)
    out.par_chunks_mut(out_features).enumerate().for_each(|(b, out_row)| {
        let x_row_start = b * in_features;
        let x_row = &x[x_row_start..x_row_start + in_features];

        // For each output neuron
        for o in 0..out_features {
            let mut sum = 0.0;
            let w_row_start = o * in_features;
            
            // Dot product: x_row . weight_row[o]
            for i in 0..in_features {
                sum += x_row[i] * weight[w_row_start + i];
            }

            if let Some(b) = bias {
                sum += b[o];
            }
            
            out_row[o] = sum;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_basic() {
        // x: 2x3
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        // w: 2x3 (transposed logic in code implies we read weight as [out, in])
        // [[0.1, 0.2, 0.3],
        //  [0.4, 0.5, 0.6]]
        //
        // y[0,0] = 1*0.1 + 2*0.2 + 3*0.3 + b[0]
        let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        // b: 2
        let b = vec![0.1, 0.2];
        
        let batch_size = 2;
        let in_features = 3;
        let out_features = 2;
        let mut out = vec![0.0; batch_size * out_features];

        linear_forward(&x, &w, Some(&b), &mut out, batch_size, in_features, out_features);

        // Expected:
        // Row 0:
        // 1*0.1 + 2*0.2 + 3*0.3 = 0.1 + 0.4 + 0.9 = 1.4. + Bias 0.1 = 1.5
        // 1*0.4 + 2*0.5 + 3*0.6 = 0.4 + 1.0 + 1.8 = 3.2. + Bias 0.2 = 3.4
        
        // Row 1:
        // 4*0.1 + 5*0.2 + 6*0.3 = 0.4 + 1.0 + 1.8 = 3.2. + Bias 0.1 = 3.3
        // 4*0.4 + 5*0.5 + 6*0.6 = 1.6 + 2.5 + 3.6 = 7.7. + Bias 0.2 = 7.9
        
        assert!((out[0] - 1.5).abs() < 1e-5);
        assert!((out[1] - 3.4).abs() < 1e-5);
        assert!((out[2] - 3.3).abs() < 1e-5);
        assert!((out[3] - 7.9).abs() < 1e-5);
    }
}
