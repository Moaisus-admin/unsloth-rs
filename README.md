# Unsloth Rust 🦥🦀

> **Unsloth, but faster. Written in Rust.**

This library implements high-performance optimization kernels (Flash Attention, RoPE, RMSNorm) and training loops for LLMs using Rust and Candle. It provides a drop-in replacement API for the original Unsloth library.

## Features

-   **Zero-Dependency Python Install**: Pre-built wheels for Linux, macOS, and Windows.
-   **High Performance**: Custom CUDA kernels and Rust-based orchestration.
-   **Native Compatibility**: Includes `unsloth_native` API that mimics `unsloth`.
-   **LoRA Support**: Efficient LoRA injection and training.
-   **GGUF Export**: Merge and save directly to Safetensors for GGUF conversion.

## Installation

### Python (Recommended)

```bash
pip install unsloth_rs
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
unsloth_rs = "0.1.0"
```

## Usage

```python
from unsloth_native import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_4bit=True
)
# ... training code ...
```

## License

Apache-2.0
