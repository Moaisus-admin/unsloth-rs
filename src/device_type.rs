use once_cell::sync::Lazy;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cuda,
    Hip,
    Xpu,
    Cpu, // Fallback, though unsloth python implies Error if not GPU
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cuda => write!(f, "cuda"),
            DeviceType::Hip => write!(f, "hip"),
            DeviceType::Xpu => write!(f, "xpu"),
            DeviceType::Cpu => write!(f, "cpu"),
        }
    }
}

// Logic to determine device type.
// In a real scenario, this would interface with Torch C++ API (tch crate) or system calls.
// For this conversion, we structure the logic and partial checks.
pub fn is_hip() -> bool {
    // Python: return bool(getattr(getattr(torch, "version", None), "hip", None))
    // Rust: We would check tch::utils::has_hip() or similar if available.
    // Placeholder: Check environment variable or typical ROCm path for now.
    std::env::var("ROCM_PATH").is_ok() || std::path::Path::new("/opt/rocm").exists()
}

pub fn get_device_type() -> Result<DeviceType, String> {
    // Python: checks torch.cuda.is_available(), etc.
    
    // Simulating torch.cuda.is_available() logic
    // In a real binding, use: tch::Cuda::is_available()
    
    // We will favor CUDA if likely present
    let cuda_available = std::process::Command::new("nvidia-smi").output().is_ok();
    
    if cuda_available {
        if is_hip() {
            return Ok(DeviceType::Hip);
        }
        return Ok(DeviceType::Cuda);
    }
    
    // Check XPU
    // Placeholder logic for XPU
    let xpu_available = std::env::var("ONEAPI_ROOT").is_ok(); // Heuristic
    if xpu_available {
        return Ok(DeviceType::Xpu);
    }

    // Check accelerator logic from Python?
    // "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
    Err("Unsloth cannot find any torch accelerator? You need a GPU.".to_string())
}

// Global cached value
pub static DEVICE_TYPE: Lazy<Result<DeviceType, String>> = Lazy::new(|| get_device_type());

pub static DEVICE_TYPE_TORCH: Lazy<Result<DeviceType, String>> = Lazy::new(|| {
    match *DEVICE_TYPE {
        Ok(dt) => {
            if dt == DeviceType::Hip {
                Ok(DeviceType::Cuda) // HIP fails for autocast, use CUDA
            } else {
                Ok(dt)
            }
        },
        Err(ref e) => Err(e.clone())
    }
});

pub fn get_device_count() -> usize {
    match *DEVICE_TYPE {
        Ok(DeviceType::Cuda) | Ok(DeviceType::Hip) => {
            // Python: torch.cuda.device_count()
            // Placeholder: Parse nvidia-smi -L | wc -l or similar
            // For now, return 1 if valid, else 0
            
            // Should implement actual count logic later
            1 
        },
        Ok(DeviceType::Xpu) => {
            1
        },
        _ => 1
    }
}

pub static DEVICE_COUNT: Lazy<usize> = Lazy::new(|| get_device_count());

pub static ALLOW_PREQUANTIZED_MODELS: Lazy<bool> = Lazy::new(|| {
    // Python logic checks bitsandbytes version and wrappers
    // Logic: bitsandbytes >= 0.49.0
    
    match *DEVICE_TYPE {
        Ok(DeviceType::Hip) => {
            // Check for bitsandbytes availability
            // This requires Python interop or verifying libraries on disk
            // Defaulting to Python's simplified logic
            false // Default safe backup
        },
        _ => true
    }
});

pub fn is_bf16_supported() -> bool {
    // Python: checks torch.cuda.is_bf16_supported() (major >= 8 for CUDA)
    // Rust placeholder:
    match *DEVICE_TYPE {
        Ok(DeviceType::Cuda) => {
            // Check compute capability if possible. 
            // For now, assume true if CUDA is present in this mocked env, or false if unknown.
            // In reality, we'd query nvidia-smi or C++ API.
            true 
        },
        Ok(DeviceType::Hip) => {
            // Check hip bf16 support
            true
        },
        Ok(DeviceType::Xpu) => {
            true
        },
        _ => false
    }
}

pub static SUPPORTS_BFLOAT16: Lazy<bool> = Lazy::new(|| is_bf16_supported());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", DeviceType::Cuda), "cuda");
        assert_eq!(format!("{}", DeviceType::Hip), "hip");
        assert_eq!(format!("{}", DeviceType::Xpu), "xpu");
        assert_eq!(format!("{}", DeviceType::Cpu), "cpu");
    }

    #[test]
    fn test_get_device_type_runs() {
        // This test mostly checks that it doesn't panic.
        // It might return Err or Ok depending on the environment.
        let result = get_device_type();
        println!("Device type detection result: {:?}", result);
    }
}
