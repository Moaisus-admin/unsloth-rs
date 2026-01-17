use super::llama::FastLlamaModel;
use super::loader::FastLanguageModel;

pub struct FastGemma2Model;

impl FastGemma2Model {
    pub fn from_pretrained(
        model_name: &str,
        max_seq_length: Option<usize>,
        dtype: Option<String>,
        load_in_4bit: bool,
    ) -> Result<FastLanguageModel, String> {
        FastLlamaModel::from_pretrained(model_name, max_seq_length, dtype, load_in_4bit)
    }
}
