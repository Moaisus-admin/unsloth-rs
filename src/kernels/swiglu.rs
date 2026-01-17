// Interface for SwiGLU kernel
// Corresponds to unsloth/kernels/swiglu.py

pub struct SwiGLU;

impl SwiGLU {
    /// CPU Reference implementation of SwiGLU
    /// f = e * sigmoid(e)
    /// h = f * g
    pub fn apply(e: &[f32], g: &[f32], out: &mut [f32]) {
        assert!(e.len() == g.len());
        assert!(e.len() == out.len());

        for i in 0..e.len() {
            let e_val = e[i];
            let g_val = g[i];
            
            // Sigmoid: 1 / (1 + exp(-x))
            let sigmoid = 1.0 / (1.0 + (-e_val).exp());
            let f_val = e_val * sigmoid;
            
            out[i] = f_val * g_val;
        }
    }
}

pub fn swiglu_forward(e: &[f32], g: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; e.len()];
    SwiGLU::apply(e, g, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_basic() {
        // Test e=0.0
        // sigmoid(0) = 0.5
        // f = 0 * 0.5 = 0
        // h = 0 * g = 0
        let e = vec![0.0];
        let g = vec![10.0];
        let mut out = vec![0.0];
        
        SwiGLU::apply(&e, &g, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        
        // Test large e (sigmoid -> 1)
        // e=10.0 => sigmoid approx 1.0
        // f = 10 * 1 = 10
        // h = 10 * 2 = 20
        let e = vec![10.0];
        let g = vec![2.0];
        SwiGLU::apply(&e, &g, &mut out);
        assert!((out[0] - 20.0).abs() < 1e-2);
        
        // Test large negative e (sigmoid -> 0)
        // e=-10 => sigmoid approx 0
        // f = -10 * 0 = 0
        // h = 0
        let e = vec![-10.0];
        let g = vec![2.0];
        SwiGLU::apply(&e, &g, &mut out);
        assert!((out[0]).abs() < 1e-2);
    }
}
