use pyo3::prelude::*;
use crate::models::llama::FastLlamaModel;
use crate::trainer::{UnslothTrainer, UnslothTrainingArguments};
use serde_json::json;
use crate::dataprep::loader::TensorDataLoader;

#[pyclass]
struct PyFastLlamaModel {
    inner: FastLlamaModel,
}

#[pymethods]
impl PyFastLlamaModel {
    #[staticmethod]
    fn from_pretrained(model_name: &str, max_seq_length: Option<usize>) -> PyResult<Self> {
        // Assume CUDA if available, else CPU.
        // For bindings, we might want to expose device selection.
        let wrapper = FastLlamaModel::from_pretrained(
            model_name,
            max_seq_length,
            None, // dtype
            false, // load_in_4bit
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        let inner_arc = wrapper.inner_model
             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No inner model found"))?;
             
        let model = inner_arc.downcast_ref::<FastLlamaModel>()
             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Inner model is not FastLlamaModel"))?
             .clone();
        
        Ok(PyFastLlamaModel { inner: model })
    }
    
    fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        self.inner.save_pretrained(save_directory)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn merge_and_save(&mut self, path: &str) -> PyResult<()> {
         // Merge
         self.inner.merge_and_unload()
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             
         // Get state dict
         let state_dict = self.inner.get_state_dict()
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             
         // Save as safetensors (Merged)
         // We can use candle implementation of safetensors saving
         candle_core::safetensors::save(&state_dict, path)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             
         Ok(())
    }

    fn add_lora_adapters(&mut self, r: usize, lora_alpha: f64, target_modules: Vec<String>) -> PyResult<()> {
         self.inner.add_lora_adapters(r, lora_alpha, target_modules)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

// Trainer Wrapper
#[pyclass]
struct PyUnslothTrainer {
    inner: UnslothTrainer,
}

#[pymethods]
impl PyUnslothTrainer {
    #[new]
    fn new(model: &PyFastLlamaModel, learning_rate: f64, steps: usize) -> PyResult<Self> {
        // Clone model to new instance (FastLlamaModel is internal, but cheap clone if VarBuilder shares data? No params are tensors.)
        // We moved ownership in rust trainer. We need to clone.
        // FastLlamaModel structs usually need to be cloneable or wrapped in Arc.
        // Assuming FastLlamaModel implements Clone (it does if Tensors do).
        
        let args = UnslothTrainingArguments {
            learning_rate,
            batch_size: 2, // hardcoded for now or exposed
            steps,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
        };
        
        // We need to clone the model because Trainer takes ownership.
        // We should ensure FastLlamaModel derives Clone.
        // Inner Tensors are RefCounted so it's cheap.
        
        match UnslothTrainer::new(model.inner.clone(), args) {
             Ok(trainer) => Ok(PyUnslothTrainer { inner: trainer }),
             Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }
    
    fn train(&mut self, input_ids: Vec<Vec<u32>>, batch_size: usize) -> PyResult<()> {
         // Create DataLoader from input_ids (pre-tokenized)
         let device = self.inner.model.device.clone();
         
         let dataset: Vec<serde_json::Value> = input_ids.into_iter()
            .map(|ids| json!({ "input_ids": ids }))
            .collect();
            
         // Create loader
         // Assuming shuffle=true for training
         let loader = TensorDataLoader::new(
             dataset,
             batch_size,
             device,
             true
         );
         
         self.inner.train(&loader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

#[pymodule]
fn unsloth_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastLlamaModel>()?;
    m.add_class::<PyUnslothTrainer>()?;
    Ok(())
}
