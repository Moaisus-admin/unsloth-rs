// Interface for GeGLU kernel
// Corresponds to unsloth/kernels/geglu.py

pub struct GeGLU;

impl GeGLU {
    /// CPU Reference implementation of GeGLU
    /// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// out = gelu(e) * g
    pub fn apply(e: &[f32], g: &[f32], out: &mut [f32]) {
        assert!(e.len() == g.len());
        assert!(e.len() == out.len());

        let sqrt_2_over_pi = 0.7978845608;
        let coef = 0.044715;

        for i in 0..e.len() {
            let x = e[i];
            let gate = g[i];
            
            // GELU approximation
            let inner = sqrt_2_over_pi * (x + coef * x.powi(3));
            let tanh_val = inner.tanh();
            let gelu_val = 0.5 * x * (1.0 + tanh_val);
            
            out[i] = gelu_val * gate;
        }
    }
}

pub fn geglu_forward(e: &[f32], g: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; e.len()];
    GeGLU::apply(e, g, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geglu_basic() {
        // Test x=0
        // gelu(0) = 0.5 * 0 * (1 + tanh(0)) = 0
        // out = 0 * g = 0
        let e = vec![0.0];
        let g = vec![10.0];
        let mut out = vec![0.0];
        GeGLU::apply(&e, &g, &mut out);
        assert!(out[0].abs() < 1e-6);
        
        // Test large positive x (approx identity)
        // gelu(x) approx x for large x
        // x=3.0 -> gelu(3) approx 2.998
        // out = 2.998 * 2 = 5.996
        
        let e = vec![3.0];
        let g = vec![2.0];
        GeGLU::apply(&e, &g, &mut out);
        // gelu(3) ~ 2.99636
        // out ~ 5.9927
        assert!((out[0] - 5.99).abs() < 1e-1);
        
        // Test large negative x (approx 0)
        // gelu(-3) approx 0
        let e = vec![-3.0];
        let g = vec![2.0];
        GeGLU::apply(&e, &g, &mut out);
        assert!(out[0].abs() < 1e-1);
    }
}
