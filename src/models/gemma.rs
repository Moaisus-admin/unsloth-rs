use super::llama::FastLlamaModel;
use super::loader::FastLanguageModel;

pub struct FastGemmaModel;

impl FastGemmaModel {
    pub fn from_pretrained(
        model_name: &str,
        max_seq_length: Option<usize>,
        dtype: Option<String>,
        load_in_4bit: bool,
    ) -> Result<FastLanguageModel, String> {
        // Gemma logic often delegates to Llama logic in Unsloth
        FastLlamaModel::from_pretrained(
            model_name,
            max_seq_length,
            dtype,
            load_in_4bit
        ).map(|mut model| {
            model.model_type = crate::models::loader::ModelType::Gemma;
            model
        })
    }
}
