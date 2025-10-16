from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
import glob, os, json

def train_16k_tokenizer(data_dir, output_dir, vocab_size=16000):
    """Train a custom GPT-2 style tokenizer with reduced vocabulary."""
    
    # Initialize a BPE tokenizer (GPT-2 style; no UNK)
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    
    # GPT-2 byte-level BPE settings
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Special tokens included in vocab by trainer
    special_tokens = ["<|endoftext|>", "[SPK]"]
    
    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    files = glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True)
    print(f"Training tokenizer on {len(files)} files...")
    
    # Train the tokenizer
    tokenizer.train(files, trainer)
    
    # Wrap in HF fast tokenizer (do NOT set unk/bos)
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    wrapped.eos_token = "<|endoftext|>"
    wrapped.pad_token = "<|endoftext|>"
    # Ensure [SPK] is registered as additional special (if not already)
    wrapped.add_special_tokens({"additional_special_tokens": ["[SPK]"]})
    wrapped.padding_side = "left"  # decoder-only models

    # Set model_max_length on the HF wrapper
    wrapped.model_max_length = 1024
    
    os.makedirs(output_dir, exist_ok=True)
    wrapped.save_pretrained(output_dir)
    print(f"Saved 16k tokenizer to {output_dir} (vocab={len(wrapped)}, max_len={wrapped.model_max_length})")
    return wrapped

if __name__ == "__main__":
    train_16k_tokenizer(
        data_dir="./data/pretrain/bnc",
        output_dir="./model_babylm_gpt2_16k",
        vocab_size=16000,
    )