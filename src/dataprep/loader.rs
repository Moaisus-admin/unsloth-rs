use std::fs::File;
use std::io::{Read, BufRead, BufReader};
use std::path::Path;
use tokenizers::Tokenizer;
use serde_json::Value;
use regex::Regex;

pub struct RawTextDataLoader {
    tokenizer: Tokenizer,
    chunk_size: usize,
    stride: usize,
    return_tokenized: bool,
    eos_token_id: Option<u32>,
}

impl RawTextDataLoader {
    pub fn new(tokenizer: Tokenizer, chunk_size: usize, stride: usize, return_tokenized: bool, eos_token_id: Option<u32>) -> Result<Self, String> {
        if chunk_size == 0 {
            return Err("chunk_size must be positive".to_string());
        }
        if stride >= chunk_size {
            return Err(format!("stride ({}) must be smaller than chunk_size ({})", stride, chunk_size));
        }
        Ok(Self {
            tokenizer,
            chunk_size,
            stride,
            return_tokenized,
            eos_token_id,
        })
    }

    pub fn detect_format(&self, file_path: &Path) -> &'static str {
        match file_path.extension().and_then(|ext| ext.to_str()) {
            Some("txt") => "plain_text",
            Some("md") => "markdown",
            Some("json") => "json_lines",
            Some("jsonl") => "json_lines",
            // Some("csv") => "csv_text_column",
            _ => "plain_text",
        }
    }

    pub fn load_from_file(&self, file_path: &Path) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let format = self.detect_format(file_path);
        let text_content = self.read_file_by_format(file_path, format)?;
        
        if text_content.trim().is_empty() {
            return Err(format!("File {:?} is empty or contains only whitespace", file_path).into());
        }

        let chunks = self.smart_chunk_text(&text_content)?;
        Ok(self.create_causal_dataset(chunks))
    }

    fn read_file_by_format(&self, file_path: &Path, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        match format {
            "plain_text" | "markdown" => {
                let mut content = String::new();
                reader.read_to_string(&mut content)?;
                Ok(content)
            }
            "json_lines" => {
                let mut content = String::new();
                for line in reader.lines() {
                    let line = line?;
                    if let Ok(json) = serde_json::from_str::<Value>(&line) {
                        if let Some(text) = self.extract_text_from_json(&json) {
                            content.push_str(&text);
                            content.push_str("\n\n");
                        }
                    }
                }
                Ok(content)
            }
            _ => {
                 let mut content = String::new();
                reader.read_to_string(&mut content)?;
                Ok(content)
            }
        }
    }

    fn extract_text_from_json(&self, data: &Value) -> Option<String> {
        let fields = ["text", "content", "message", "body", "description", "prompt"];
        for field in fields {
            if let Some(val) = data.get(field) {
                if let Some(s) = val.as_str() {
                    return Some(s.to_string());
                }
            }
        }
        None
    }

    pub fn smart_chunk_text(&self, text: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> { // Returns list of chunk objects (maps) or strings
        let encoding = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        let tokens = encoding.get_ids();
        let len = tokens.len();

        let mut chunks = Vec::new();
        
        if len <= self.chunk_size {
             if self.return_tokenized {
                 let mut chunk_tokens = tokens.to_vec();
                 if let Some(eos) = self.eos_token_id {
                     chunk_tokens.push(eos);
                 }
                 
                 let attention_mask = vec![1; chunk_tokens.len()];
                 
                 let chunk_map = serde_json::json!({
                     "input_ids": chunk_tokens,
                     "attention_mask": attention_mask
                 });
                 chunks.push(chunk_map);
             } else {
                 chunks.push(Value::String(text.to_string()));
             }
             return Ok(chunks);
        }
        
        let mut start_idx = 0;
        while start_idx < len {
            let end_idx = std::cmp::min(start_idx + self.chunk_size, len);
            let mut chunk_tokens = tokens[start_idx..end_idx].to_vec();
            
             if self.return_tokenized {
                 if end_idx == len || chunk_tokens.len() == self.chunk_size {
                     if let Some(eos) = self.eos_token_id {
                         chunk_tokens.push(eos);
                     }
                 }

                 let attention_mask = vec![1; chunk_tokens.len()];
                 let chunk_map = serde_json::json!({
                     "input_ids": chunk_tokens,
                     "attention_mask": attention_mask
                 });
                 chunks.push(chunk_map);
             } else {
                 let chunk_text = self.tokenizer.decode(&chunk_tokens, true).map_err(|e| e.to_string())?;
                 chunks.push(Value::String(chunk_text));
             }
             
             if end_idx == len {
                 break;
             }
             start_idx += self.chunk_size - self.stride;
        }

        Ok(chunks)
    }

    fn create_causal_dataset(&self, chunks: Vec<Value>) -> Vec<Value> {
        // Just return chunks for now, they are properly formatted as dicts or strings
        chunks
    }
}

pub struct TextPreprocessor;

impl TextPreprocessor {
    pub fn clean_text(text: &str) -> String {
        let re_space = Regex::new(r"\s+").unwrap();
        let re_chars = Regex::new(r"[^\x20-\x7E\n\t]").unwrap();
        let re_newline = Regex::new(r"\n{3,}").unwrap();

        let text = re_space.replace_all(text, " ");
        let text = re_chars.replace_all(&text, "");
        let text = text.replace("\r\n", "\n").replace("\r", "\n");
        let text = re_newline.replace_all(&text, "\n\n");
        
        text.trim().to_string()
    }

    pub fn extract_sections(text: &str, patterns: &[&str]) -> Vec<String> {
        let mut sections = Vec::new();
        for pattern in patterns {
             if let Ok(re) = Regex::new(pattern) {
                 for cap in re.captures_iter(text) {
                     if let Some(m) = cap.get(0) {
                         sections.push(m.as_str().to_string());
                     }
                 }
             }
        }
        sections
    }

    pub fn add_structure_tokens(text: &str) -> String {
        // Rust Regex has (?m) for multiline mode.
        let re_chapter = Regex::new(r"(?m)^# (.+)$").unwrap();
        let text = re_chapter.replace_all(text, "<|chapter|>$1<|/chapter|>").to_string();

        let re_section = Regex::new(r"(?m)^## (.+)$").unwrap();
        let text = re_section.replace_all(&text, "<|section|>$1<|/section|>").to_string();

        let re_subsection = Regex::new(r"(?m)^### (.+)$").unwrap();
        let text = re_subsection.replace_all(&text, "<|subsection|>$1<|/subsection|>").to_string();
        
        // Code blocks: Use (?s) for dot matches newline.
        let re_code = Regex::new(r"(?s)```(\w*)\n(.*?)\n```").unwrap();
        let text = re_code.replace_all(&text, "<|code|$1|>$2<|/code|>").to_string();
        
        text
    }
    
    pub fn validate_dataset(dataset_texts: &[String]) -> serde_json::Value {
        let total = dataset_texts.len();
        let mut empty = 0;
        let mut min_len = usize::MAX;
        let mut max_len = 0;
        let mut sum_len = 0;
        
        for text in dataset_texts {
            if text.trim().is_empty() {
                empty += 1;
                continue;
            }
            let len = text.len();
            if len < min_len { min_len = len; }
            if len > max_len { max_len = len; }
            sum_len += len;
        }
        
        let avg_len = if total > 0 { sum_len as f64 / total as f64 } else { 0.0 };
        if min_len == usize::MAX { min_len = 0; }

        serde_json::json!({
            "total_samples": total,
            "empty_samples": empty,
            "min_length": min_len,
            "max_length": max_len,
            "avg_length": avg_len
        })
    }
}

use candle_core::{Tensor, Device};

pub struct TensorDataLoader {
    dataset: Vec<Value>,
    batch_size: usize,
    device: Device,
    shuffle: bool,
}

impl TensorDataLoader {
    pub fn new(dataset: Vec<Value>, batch_size: usize, device: Device, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            device,
            shuffle,
        }
    }

    pub fn iter(&self) -> TensorBatchIterator<'_> {
        TensorBatchIterator {
            dataset: &self.dataset,
            batch_size: self.batch_size,
            device: &self.device,
            current_idx: 0,
            indices: (0..self.dataset.len()).collect(), // TODO: Implement shuffle
        }
    }
}

pub struct TensorBatchIterator<'a> {
    dataset: &'a Vec<Value>,
    batch_size: usize,
    device: &'a Device,
    current_idx: usize,
    indices: Vec<usize>,
}

impl<'a> Iterator for TensorBatchIterator<'a> {
    type Item = Result<(Tensor, Tensor), Box<dyn std::error::Error>>; // (input_ids, labels)

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.dataset.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        self.current_idx = end_idx;

        let mut batch_inputs = Vec::with_capacity(batch_indices.len());
        
        for &idx in batch_indices {
            let item = &self.dataset[idx];
            // Expect item to be a dict with input_ids or just raw tokens?
            // RawTextDataLoader output format depends on flags.
            // Assuming "input_ids" key exists if it's a map, or parsing string?
            // "smart_chunk_text" with return_tokenized=true returns dict: { "input_ids": [...], ... }
            
            let ids: Vec<u32> = if let Some(obj) = item.as_object() {
                if let Some(vals) = obj.get("input_ids") {
                    serde_json::from_value(vals.clone()).ok()?
                } else {
                    return Some(Err("Missing input_ids in dataset item".into()));
                }
            } else {
                return Some(Err("Dataset item is not a JSON object".into()));
            };
            
            batch_inputs.push(ids);
        }

        // Pad batch if lengths differ? 
        // For now assume chunks are equal length (chunk_size). 
        // Logic for padding inputs if needed?
        // Let's assume uniform size for simplicity or implement simple padding.
        let max_len = batch_inputs.iter().map(|v| v.len()).max().unwrap_or(0);
        let mut flat_input = Vec::with_capacity(batch_inputs.len() * max_len);
        
        // Pad with 0 (eos?) or 0?
        for sample in batch_inputs {
            let mut s = sample.clone();
            s.resize(max_len, 0); 
            flat_input.extend(s);
        }

        let tensor = match Tensor::from_vec(flat_input, (batch_indices.len(), max_len), self.device) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        
        // Causal LM: Labels same as inputs usually
        let labels = tensor.clone();

        Some(Ok((tensor, labels)))
    }
}
