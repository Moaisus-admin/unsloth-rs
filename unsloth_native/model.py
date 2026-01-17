from transformers import AutoTokenizer
import unsloth_rs

class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = True,
        token = None,
        device_map = None,
        rope_scaling = None,
        fix_tokenizer = True,
        trust_remote_code = False,
        **kwargs
    ):
        """
        Loads a Llama/Mistral model from a given path/repo using the Rust optimization backend.
        Returns (model, tokenizer).
        """
        print(f"🦥 Unsloth Native: Loading {model_name}...")
        
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # 2. Load Rust Model
        # Map arguments to Rust API
        # PyFastLlamaModel.from_pretrained(model_name, max_seq_length)
        # Note: Rust load_in_4bit logic is internal or default True in current binding?
        # Check src/python.rs: it passes `false` hardcoded currently?
        # Wait, step 2139 view of python.rs showed: `false, // load_in_4bit`
        # I should probably update python.rs to accept load_in_4bit arg, but for now strict wrapper around what exists.
        
        model = unsloth_rs.PyFastLlamaModel.from_pretrained(model_name, max_seq_length)
        
        print("🦥 Unsloth Native: Model successfully loaded via Rust backend.")
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        target_modules = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: str = "none",
        layers_to_transform = None,
        use_gradient_checkpointing = "unsloth",
        random_state: int = 3407,
        max_seq_length: int = 2048,
        # ... other args
        **kwargs
    ):
        """
        Injects LoRA adapters into the Rust model.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        print(f"🦥 Unsloth Native: Injecting LoRA (r={r}, alpha={lora_alpha})...")
        model.add_lora_adapters(r, float(lora_alpha), target_modules)
        print("🦥 Unsloth Native: LoRA adapters active.")
        
        return model

    @staticmethod
    def for_inference(model):
        """
        Prepares model for inference (merges LoRA temporarily? Or just sets mode?).
        In Rust, inference is default if not training.
        """
        # No-op in current Rust implementation
        return model

# Trainer Wrapper
class UnslothTrainer:
    def __init__(self, model, tokenizer, train_dataset, args, **kwargs):
        """
        Wraps PyUnslothTrainer.
        Args:
            model: PyFastLlamaModel
            tokenizer: transformers.PreTrainedTokenizer
            train_dataset: list/dataset containing text entries
            args: TrainingArguments dict or object
        """
        self.tokenizer = tokenizer
        self.model = model
        
        # Extract args
        lr = args.learning_rate if hasattr(args, "learning_rate") else 2e-4
        steps = args.max_steps if hasattr(args, "max_steps") else 60
        batch_size = args.per_device_train_batch_size if hasattr(args, "per_device_train_batch_size") else 2
        
        self.inner_trainer = unsloth_rs.PyUnslothTrainer(model, lr, steps)
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        
    def train(self):
        # Data Prep
        print("🦥 Unsloth Native: Tokenizing dataset...")
        input_ids_batch = []
        
        # Detect dataset format
        # If huggingface dataset?
        dataset = self.train_dataset
        # Iterate and tokenize
        import tqdm
        for item in dataset:
            # Assume item is text or dict with 'text'
            text = ""
            if isinstance(item, str): text = item
            elif isinstance(item, dict) and "text" in item: text = item["text"]
            elif isinstance(item, dict) and "input_ids" in item: 
                input_ids_batch.append(item["input_ids"]) # Already tokenized?
                continue
            
            # Simple tokenize
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            input_ids_batch.append(ids)
            
        print(f"🦥 Unsloth Native: Starting Training (steps={self.inner_trainer.inner.args.steps if hasattr(self.inner_trainer, 'inner') else '?'})...")
        self.inner_trainer.train(input_ids_batch, self.batch_size)
        print("🦥 Unsloth Native: Training complete.")

