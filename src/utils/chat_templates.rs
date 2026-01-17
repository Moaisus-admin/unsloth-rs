use std::collections::HashMap;
use once_cell::sync::Lazy;

// Structure to hold chat template information
pub struct ChatTemplate {
    pub template: &'static str,
    pub eos_token: &'static str,
    pub use_bos: bool, // Simplified for now
}

// Unsloth template
pub const UNSLOTH_TEMPLATE: &str = r#"{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% set loop_messages = messages[1:] %}{% else %}{{ '{system_message}' + '\n' }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '>>> User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '>>> Assistant: ' }}{% endif %}"#;

// Zephyr template
pub const ZEPHYR_TEMPLATE: &str = r#"{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}{% else %}{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"#;

// ChatML template
pub const CHATML_TEMPLATE: &str = r#"{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% else %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"#;

pub static CHAT_TEMPLATES: Lazy<HashMap<&'static str, ChatTemplate>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("unsloth", ChatTemplate {
        template: UNSLOTH_TEMPLATE,
        eos_token: "eos_token",
        use_bos: false,
    });
    m.insert("zephyr", ChatTemplate {
        template: ZEPHYR_TEMPLATE,
        eos_token: "eos_token",
        use_bos: false,
    });
    m.insert("chatml", ChatTemplate {
        template: CHATML_TEMPLATE,
        eos_token: "<|im_end|>",
        use_bos: true,
    });
    // Add Llama3, Mistral, etc. as needed
    m
});

pub fn get_chat_template(name: &str) -> Option<&'static ChatTemplate> {
    CHAT_TEMPLATES.get(name)
}
