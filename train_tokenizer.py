from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
import glob
import os

def train_16k_tokenizer(data_dir, output_dir, vocab_size=16000):
    """Train a custom GPT-2 style tokenizer with reduced vocabulary."""
    
    # Initialize a BPE tokenizer (same as GPT-2)
    tokenizer = Tokenizer(models.BPE())
    
    # Use GPT-2's preprocessing (ByteLevel)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Special tokens
    special_tokens = ["<|endoftext|>", "[SPK]"]
    
    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Collect training files
    pattern = os.path.join(data_dir, "**/*.md")
    files = glob.glob(pattern, recursive=True)
    print(f"Training tokenizer on {len(files)} files...")
    
    # Train the tokenizer
    tokenizer.train(files, trainer)
    
    # Add post-processor (for special tokens)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Wrap in HuggingFace tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    
    # Save
    wrapped_tokenizer.save_pretrained(output_dir)
    print(f"Saved 16k tokenizer to {output_dir}")
    print(f"Final vocab size: {len(wrapped_tokenizer)}")
    
    return wrapped_tokenizer

if __name__ == "__main__":
    train_16k_tokenizer(
        data_dir="./data/pretrain/bnc",
        output_dir="./model_babylm_gpt2_16k",
        vocab_size=16000
    )