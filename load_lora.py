import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_adapter_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def load_and_merge_lora(adapter_dir: str, base_model_name: str | None = None, dtype: str = "float32"):
    # Resolve dtype
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower(), torch.float32)

    # Detect base model from adapter config if not provided
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")

    adapter_cfg = load_adapter_config(cfg_path)
    if not base_model_name:
        base_model_name = adapter_cfg.get("base_model_name_or_path", None)
        if not base_model_name:
            raise ValueError("base_model_name_or_path missing in adapter_config.json and no --base_model provided.")

    # Sanity: verify adapter weights exist
    has_bin = os.path.exists(os.path.join(adapter_dir, "adapter_model.bin"))
    has_safe = os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))
    if not (has_bin or has_safe):
        raise FileNotFoundError(f"Adapter weights not found in {adapter_dir} (expected adapter_model.bin or adapter_model.safetensors)")

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch_dtype)

    # Load LoRA into base (PEFT wrapper)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    peft_model.eval()

    # Merge LoRA weights into base and unload PEFT wrapper
    merged = peft_model.merge_and_unload()  # returns a plain transformers model
    merged.eval()

    return merged, tokenizer

def save_merged(model, tokenizer, output_dir: str, safe_serialization: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Load GPT-2 + LoRA adapters, merge, and save HF model for lm_eval.")
    parser.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter folder (contains adapter_config.json and adapter_model.*)")
    parser.add_argument("--output_dir", required=True, help="Where to save the merged model")
    parser.add_argument("--base_model", default=None, help="Optional override for base model name (else read from adapter_config.json)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], help="Load/merge dtype")
    parser.add_argument("--test_prompt", default="Once upon a time", help="Optional quick gen test prompt")
    args = parser.parse_args()

    model, tok = load_and_merge_lora(args.adapter_dir, args.base_model, args.dtype)
    save_merged(model, tok, args.output_dir, safe_serialization=True)

    # Quick smoke test
    inputs = tok.encode(args.test_prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(inputs, max_new_tokens=32, do_sample=False)
    print(tok.decode(out_ids[0], skip_special_tokens=True))

    print(f"\nSaved merged model to: {args.output_dir}")
    print("Use with lm_eval:")
    print(f"  lm_eval --model hf --model_args pretrained={args.output_dir} --tasks hellaswag --device cuda:0")

if __name__ == "__main__":
    main()