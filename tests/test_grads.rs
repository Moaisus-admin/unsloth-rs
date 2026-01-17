
#[cfg(test)]
mod tests {
    use candle_core::{Tensor, Device, Var};
    
    #[test]
    fn test_grad_store_usage() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32], &device)?;
        let x = var.as_tensor();
        let y = (x * 2.0)?;
        
        let mut grads = y.backward()?;
        
        if let Some(g) = grads.get(&var) {
            println!("Gradient found: {:?}", g);
        }

        // Fuzzing API: check if we can insert or merge
        grads.insert(&var, x.clone()); // Uncomment to check if insert exists
        
        // Check if Tensor has backward_params or backward_with options?
        
        Ok(())
    }
}
