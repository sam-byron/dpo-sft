import torch
import json
import os
import itertools
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist

# Add a PG-safe barrier that pins the device id for NCCL
def safe_wait_for_everyone(accelerator, note=""):
    try:
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                # Pin device mapping for NCCL to avoid hangs/warnings
                dist.barrier(device_ids=[dev])
            else:
                dist.barrier()
        else:
            accelerator.wait_for_everyone()
    except TypeError:
        # Older torch without device_ids kwarg
        dist.barrier()
    except Exception as e:
        if accelerator is not None and getattr(accelerator, "is_main_process", True):
            print(f"[safe_barrier] fallback at {note}: {e}")

def generate_special_tokens_map(tokenizer, checkpoint_path, accelerator=None, C=None):
    """
    Auto-generate special_tokens_map.json for AutoTokenizer compatibility.
    
    Args:
        tokenizer: The tokenizer to extract special tokens from
        checkpoint_path: Directory where to save the special_tokens_map.json file
        accelerator: Optional accelerator instance for distributed training
        C: Optional color class for formatted output
    
    Returns:
        dict: The generated special tokens mapping
    """
    if C is None:
        # Fallback color class if none provided
        class C:
            RESET = "\033[0m"
            CYAN = "\033[36m"
    
    # Only generate on main process if using accelerator
    if accelerator is not None and not accelerator.is_main_process:
        return {}
    
    special_tokens_map = {}
    
    # Extract special tokens from the tokenizer
    vocab = tokenizer.get_vocab()
    
    # Common BERT special token patterns - adjust based on your tokenizer
    special_token_patterns = {
        'cls_token': ['[CLS]', '<cls>', '<CLS>'],
        'sep_token': ['[SEP]', '<sep>', '<SEP>'],
        'pad_token': ['[PAD]', '<pad>', '<PAD>'],
        'unk_token': ['[UNK]', '<unk>', '<UNK>'],
        'mask_token': ['[MASK]', '<mask>', '<MASK>'],
        'bos_token': ['[BOS]', '<bos>', '<BOS>', '[CLS]'],  # Often CLS serves as BOS
        'eos_token': ['[EOS]', '<eos>', '<EOS>', '[SEP]']   # Often SEP serves as EOS
    }
    
    # Find special tokens in vocabulary
    for token_type, patterns in special_token_patterns.items():
        for pattern in patterns:
            if pattern in vocab:
                special_tokens_map[token_type] = pattern
                break
    
    # Ensure directory exists
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save special_tokens_map.json
    special_tokens_path = os.path.join(checkpoint_path, "special_tokens_map.json")
    with open(special_tokens_path, "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    
    print(f"{C.CYAN}Auto-generated special_tokens_map.json with: {special_tokens_map}{C.RESET}")
    
    return special_tokens_map

# ===== Simple ANSI color helper =====
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"

# Training utilities to simplify the main training loop

def log_first_batch_info(accelerator, batch, tokenizer, loss, C):
    """Log detailed information about the first batch for debugging."""
    accelerator.print(f"{C.GREEN}Starting first batch processing... Batch keys: {list(batch.keys())}{C.RESET}")
    accelerator.print(f"{C.CYAN}Batch shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}, labels: {batch['labels'].shape}{C.RESET}")
    accelerator.print(f"{C.GREEN}Model forward pass completed, loss: {loss}{C.RESET}")
    
    # Debug: Check how many tokens are being masked
    masked_tokens = (batch["labels"] != -100).sum().item()
    total_tokens = batch["labels"].numel()
    masking_ratio = masked_tokens / total_tokens
    accelerator.print(f"{C.CYAN}Masked tokens: {masked_tokens}/{total_tokens} ({masking_ratio:.1%}){C.RESET}")
    
    # Check label distribution
    unique_labels = torch.unique(batch["labels"][batch["labels"] != -100])
    accelerator.print(f"{C.CYAN}Unique masked labels: {len(unique_labels)} (vocab size: {tokenizer.vocab_size}){C.RESET}")
    
    # Sample some of the actual tokens being masked
    sample_labels = batch["labels"][batch["labels"] != -100][:20]  # First 20 masked tokens
    sample_tokens = [tokenizer.convert_ids_to_tokens([int(label)]) for label in sample_labels if int(label) < tokenizer.vocab_size]
    accelerator.print(f"{C.CYAN}Sample masked tokens: {sample_tokens[:10]}{C.RESET}")
    
    # Check the full input to see vocabulary diversity
    all_input_tokens = torch.unique(batch["input_ids"])
    accelerator.print(f"{C.CYAN}Unique tokens in input_ids: {len(all_input_tokens)}/{tokenizer.vocab_size}{C.RESET}")

    # Sample some input tokens to see what we're working with
    sample_input_ids = batch["input_ids"][0]
    # Print
    sample_input_tokens = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in sample_input_ids if int(tid) < tokenizer.vocab_size]
    accelerator.print(f"{C.CYAN}Sample input tokens: {sample_input_tokens[:100]}{C.RESET}")

    num_unk_in_sample = (sample_input_ids == tokenizer.convert_tokens_to_ids("[UNK]")).sum().item()
    accelerator.print(f"{C.GREEN}Sample input has {num_unk_in_sample}/{len(sample_input_ids)} unknown tokens ({num_unk_in_sample/len(sample_input_ids):.1%}){C.RESET}")


def forward_pass(model, batch):
    """Perform model forward pass with proper attention mask conversion."""
    # Do NOT invert here; model handles conversion/inversion internally.
    # Ensure tensor exists and is the standard 1 (real) / 0 (pad) format.
    # The model will convert to bool and invert to produce a padding mask.
    # If a bool mask is provided, it's still fine (the model calls .bool() then ~).
    
    prediction = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return prediction.loss, prediction.logits


def gradient_step(accelerator, model, optimizer, scheduler, monitor, config, step, loss, is_main, C):
    """Handle gradient clipping, monitoring, and optimizer step."""
    if not accelerator.sync_gradients:
        return float(loss.detach())
    
    # Calculate gradient norm for monitoring
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    # Get current learning rate
    current_lr = scheduler.get_last_lr()[0]
    
    # Update monitor with current metrics
    monitor.update(float(loss.detach()), total_norm, current_lr, step)
    
    # Check for training issues
    spike_detected, spike_msg = monitor.check_loss_spike()
    oscillation_detected, osc_msg = monitor.check_oscillation()
    explosion_detected, exp_msg = monitor.check_gradient_explosion(config.get("max_grad_norm", 1.0))
    
    # Print warnings for issues
    if spike_detected and is_main:
        print(f"{C.YELLOW}⚠️  {spike_msg}{C.RESET}")
    if oscillation_detected and is_main:
        print(f"{C.YELLOW}⚠️  {osc_msg}{C.RESET}")
    if explosion_detected and is_main:
        print(f"{C.RED}🚨 {exp_msg}{C.RESET}")
    
    # Clip gradients
    accelerator.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
    
    # Check for NaN gradients
    has_nan = any(
        param.grad is not None and torch.isnan(param.grad).any()
        for param in model.parameters()
    )
    
    if has_nan:
        if is_main:
            print(f"{C.RED}Skipping optimizer step due to NaN gradients{C.RESET}")
        optimizer.zero_grad()
        return float(loss.detach())
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Return gathered loss
    return accelerator.gather(loss).mean()


def update_progress_bars(batch_bar, loss_bar, monitor, loss_value, avg_loss, step, log_steps, 
                        processed_batches, epoch_start, total_batches, is_main):
    """Update progress bars with current metrics."""
    if not is_main or step % log_steps != 0:
        return
    
    batch_bar.update(log_steps)
    loss_bar.update(1)
    
    # Add monitoring stats to loss bar
    monitor_stats = monitor.get_stats()
    loss_bar.set_postfix({
        "loss": f"{loss_value:.4f}",
        "avg_loss": f"{avg_loss:.4f}",
        "monitor": f"S:{monitor.spike_count} O:{monitor.oscillation_count} E:{monitor.explosion_count}"
    })
    
    # Print detailed monitor stats every 50 steps
    if step % (log_steps * 50) == 0:
        print(f"[Monitor] {monitor_stats}")
    
    # Update batch bar with ETA
    import time
    import datetime
    elapsed = time.time() - epoch_start
    batches_per_sec = processed_batches / elapsed if elapsed > 0 else 0
    remaining = max(total_batches - processed_batches, 0)
    eta_sec = remaining / batches_per_sec if batches_per_sec > 0 else 0
    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
    batch_bar.set_postfix({
        "batch/s": f"{batches_per_sec:.0f}",
        "ETA": eta_str,
    })

def save_once(accelerator, tokenizer, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_t = None
    if accelerator.is_main_process:
        import time as _time
        start_t = _time.time()
        print(f"{C.CYAN}[HF Save] Preparing tokenizer, modeling, and config files for save to {out}{C.RESET}")
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "save_pretrained"):
                if accelerator.is_main_process:
                    print(f"{C.DIM}[HF Save] tokenizer.save_pretrained(...){C.RESET}")
                tokenizer.save_pretrained(out)
            elif hasattr(tokenizer, "save"):
                # tokenizers.Tokenizer or custom tokenizer with .save(path)
                tok_path = out / "tokenizer.json"
                if accelerator.is_main_process:
                    print(f"{C.DIM}[HF Save] tokenizer.save('{tok_path}') (fast tokenizer){C.RESET}")
                tokenizer.save(str(tok_path))
            else:
                # Best-effort: try common attribute names
                tok_path = out / "tokenizer.json"
                if hasattr(tokenizer, "to_str"):
                    content = tokenizer.to_str()
                    with open(tok_path, "w") as f:
                        f.write(content)
                else:
                    # No known save method; skip silently
                    pass
            # Ensure special_tokens_map.json exists for AutoTokenizer
            try:
                if accelerator.is_main_process:
                    print(f"{C.DIM}[HF Save] generate_special_tokens_map(...) -> special_tokens_map.json{C.RESET}")
                generate_special_tokens_map(tokenizer, str(out), accelerator=accelerator, C=C)
            except Exception:
                pass
        except Exception as e:
            if accelerator.is_main_process:
                print(f"{C.YELLOW}[HF Save] Warning: failed to save tokenizer: {e}{C.RESET}")
    if accelerator.is_main_process:
        import time as _time
        dur = ( _time.time() - start_t ) if start_t is not None else 0.0
        print(f"[HF Save] Wrote checkpoint to {out} (safe_serialization=False; allows shared tensors) in {dur:.2f}s")
        # Also copy the current modeling/config files so trust_remote_code picks up latest class definitions
        try:
            import shutil as _shutil
            src_model = Path(__file__).resolve().parent / "modeling_ltgbert.py"
            src_conf  = Path(__file__).resolve().parent / "configuration_ltgbert.py"
            if src_model.exists():
                _shutil.copy2(str(src_model), str(out))
            if src_conf.exists():
                _shutil.copy2(str(src_conf), str(out))
            print(f"{C.DIM}[HF Save] Copied modeling/configuration files into checkpoint dir{C.RESET}")
        except Exception as _e:
            print(f"{C.YELLOW}[HF Save] Warning: failed to copy modeling/config files: {_e}{C.RESET}")

def save_hf_checkpoint(accelerator, model, tokenizer, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    to_save = accelerator.unwrap_model(model)
    # Ensure tied weights are consistent prior to save
    if hasattr(to_save, "tie_weights"):
        try:
            to_save.tie_weights()
        except Exception:
            pass
    # Register for auto class so AutoModel* can resolve this custom class from the checkpoint
    try:
        # Ensure config knows how to load as AutoConfig
        if hasattr(to_save, "config") and hasattr(to_save.config, "register_for_auto_class"):
            try:
                to_save.config.register_for_auto_class("AutoConfig")
            except TypeError:
                # Fallback to no-arg variant if older Transformers
                to_save.config.register_for_auto_class()
        # Register this class for AutoModel and/or AutoModelForMaskedLM depending on availability
        if hasattr(to_save, "register_for_auto_class"):
            try:
                # Prefer the specific masked LM auto class if applicable
                to_save.register_for_auto_class("AutoModelForMaskedLM")
            except TypeError:
                # Fallback to generic AutoModel registration or no-arg variant
                try:
                    to_save.register_for_auto_class("AutoModel")
                except Exception:
                    to_save.register_for_auto_class()
    except Exception:
        pass
    # Use safe_serialization=False to allow shared tensors (e.g., tied embeddings, custom heads)
    # This writes pytorch_model.bin and is robust to shared parameters.
    if accelerator.is_main_process:
        print(f"{C.DIM}[HF Save] save_pretrained(..., safe_serialization=False)...{C.RESET}")
    to_save.save_pretrained(out, safe_serialization=False)
    

def save_model_official_way(model, model_config, tokenizer, checkpoint_path):
    """
    Save model the official HuggingFace way for compatibility with lm_eval and other tools.
    """
    print(f"{C.CYAN}Saving model the official HuggingFace way to {checkpoint_path}{C.RESET}")
    
    # Ensure directory exists
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Register for auto class - this is the key part!
    model_config.register_for_auto_class()
    model.register_for_auto_class()  # No arguments needed for models
    
    # Save model and config using HF methods
    # Use safe_serialization=False to handle shared tensors
    model.save_pretrained(checkpoint_path, safe_serialization=False)
    model_config.save_pretrained(checkpoint_path)
    
    # Generate special_tokens_map.json for AutoTokenizer compatibility
    generate_special_tokens_map(tokenizer, checkpoint_path, C=C)
    
    print(f"{C.BOLD}{C.GREEN}Model saved the official HuggingFace way! Ready for lm_eval with trust_remote_code=True{C.RESET}")

def save_checkpoint(accelerator, model, tokenizer, checkpoint_path, step, save_steps, is_main, C, epoch=None):
    """Mid-epoch lightweight save every save_steps on rank 0 only, without collectives.

    This avoids distributed deadlocks caused by unsynchronized barriers. End-of-epoch
    saves (save_epoch_checkpoint) still perform a synchronized full state save.
    """
    # Only act on exact boundaries and not at step 0
    if step % save_steps != 0 or step == 0:
        return

    if not is_main:
        return  # non-main ranks skip IO and continue training

    try:
        print(f"{C.CYAN}[Checkpoint] (light) Starting mid-epoch save at step {step}...{C.RESET}")
        import time
        t0 = time.time()

        # Lightweight: write HF-compatible files only (model + tokenizer + auto_map)
        print(f"{C.DIM}[Checkpoint] -> save_hf_checkpoint(...){C.RESET}")
        save_hf_checkpoint(accelerator, model, tokenizer, checkpoint_path)

        # Persist training position within the epoch for resume
        if epoch is not None:
            print(f"{C.DIM}[Checkpoint] -> save_training_state(...){C.RESET}")
            save_training_state(checkpoint_path, epoch, step, is_main)

        total = time.time() - t0
        print(f"{C.GREEN}[Checkpoint] (light) Saved HF files at step {step} in {total:.2f}s{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}[Checkpoint] Warning: mid-epoch save failed at step {step}: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
    # accelerator.wait_for_everyone()


def save_training_state(checkpoint_path, epoch, step=None, is_main=True):
    """Save current training state (epoch, step) to resume later."""
    # if not is_main:
    #     return
    
    try:
        training_state = {
            "epoch": epoch,
            "step": step if step is not None else 0,
            "completed_epochs": epoch  # For clarity
        }
        
        state_file = os.path.join(checkpoint_path, "training_state.json")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(training_state, f, indent=2)
            
    except Exception as e:
        print(f"Warning: failed to save training state: {e}")


def load_training_state(checkpoint_path):
    """Load saved training state (epoch, step) to resume from."""
    try:
        state_file = os.path.join(checkpoint_path, "training_state.json")
        
        if not os.path.exists(state_file):
            return 0, 0  # Default: start from epoch 0, step 0
            
        with open(state_file, 'r') as f:
            training_state = json.load(f)
            
        epoch = training_state.get("epoch", 0)
        step = training_state.get("step", 0)
        
        return epoch, step
        
    except Exception as e:
        print(f"Warning: failed to load training state: {e}")
        return 0, 0  # Default: start from epoch 0, step 0


def save_epoch_checkpoint(accelerator, model, tokenizer, checkpoint_path, epoch, is_main, C):
    """Save checkpoint at the end of an epoch with distributed synchronization to avoid hangs."""
    # Synchronize all ranks so they reach this point together
    # safe_wait_for_everyone(accelerator, note="save_epoch_checkpoint start")
    try:
        # if is_main:

        accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
        accelerator.save_model(model, checkpoint_path)
        save_hf_checkpoint(accelerator, model, tokenizer, checkpoint_path)
        save_training_state(checkpoint_path, epoch + 1, step=0, is_main=is_main)
    except Exception as e:
        if is_main:
            print(f"{C.YELLOW}[End-of-Epoch] Warning: failed to save checkpoint: {e}{C.RESET}")
            import traceback
            traceback.print_exc()
    finally:
        # Ensure IO completes before any rank continues
        # safe_wait_for_everyone(accelerator, note="save_epoch_checkpoint end")
        if is_main:
            print(f"{C.DIM}[End-of-Epoch] All ranks passed post-save barrier{C.RESET}")


def build_batch_iterator_with_skip(train_loader, start_step, is_main, log_steps, C):
    """
    Build an iterator over the training DataLoader that fast-skips to a given batch index.
    Fast path: wraps the underlying batch_sampler with a SkipBatchSampler to avoid loading skipped batches.
    Fallback: consumes and discards batches from the original iterator.

    Returns (batch_iterator, actual_start_index) where the iterator yields (step, batch)
    starting at actual_start_index, and step numbers are aligned to global logic.
    """
    try:
        total_batches = len(train_loader)
    except TypeError:
        total_batches = None

    if start_step <= 0:
        if is_main:
            if total_batches is not None:
                print(f"{C.CYAN}Processing all {total_batches} batches normally (no skipping){C.RESET}")
            else:
                print(f"{C.CYAN}Processing all batches normally (no skipping){C.RESET}")
        return enumerate(train_loader), 0

    if total_batches is not None:
        print(f"{C.YELLOW}Skipping {start_step} batches to resume...{C.RESET}")
        print(f"{C.CYAN}Dataloader has {total_batches} total batches{C.RESET}")
        if start_step >= total_batches:
            print(f"{C.RED}Error: Cannot skip to step {start_step}, dataloader only has {total_batches} batches{C.RESET}")
            print(f"{C.YELLOW}This suggests the checkpoint is invalid or from a different dataset configuration{C.RESET}")
            print(f"{C.CYAN}Falling back to starting from step 0{C.RESET}")
            return enumerate(train_loader), 0

    # Fast path using a SkipBatchSampler
    try:
        class SkipBatchSampler(Sampler):
            def __init__(self, base_batch_sampler, skip_batches: int):
                self.base = base_batch_sampler
                self.skip = max(0, int(skip_batches))
            def __iter__(self):
                return itertools.islice(iter(self.base), self.skip, None)
            def __len__(self):
                try:
                    return max(0, len(self.base) - self.skip)
                except TypeError:
                    return 0

        base_bs = getattr(train_loader, "batch_sampler", None)
        if base_bs is None:
            raise RuntimeError("train_loader has no batch_sampler; cannot build fast remainder loader")

        remainder_bs = SkipBatchSampler(base_bs, start_step)

        # Build DataLoader kwargs mirroring the original where possible
        dl_kwargs = dict(
            dataset=train_loader.dataset,
            batch_sampler=remainder_bs,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
            collate_fn=getattr(train_loader, "collate_fn", None),
        )
        if hasattr(train_loader, "persistent_workers"):
            dl_kwargs["persistent_workers"] = getattr(train_loader, "persistent_workers", False)
        # Only set prefetch_factor when it exists and workers > 0 (PyTorch requires workers > 0)
        if getattr(train_loader, "num_workers", 0) > 0 and hasattr(train_loader, "prefetch_factor"):
            dl_kwargs["prefetch_factor"] = getattr(train_loader, "prefetch_factor")

        remainder_loader = DataLoader(**dl_kwargs)

        if is_main and total_batches is not None:
            # Progress bars: mark skipped progress visually
            # batch_bar.update(start_step)
            # loss_bar.update(start_step // max(1, log_steps))
            # loss_bar.set_postfix({"status": f"skipped to step {start_step} (fast)"})
            remaining_batches = total_batches - start_step
            print(f"{C.GREEN}Fast skip enabled: starting at batch {start_step} with {remaining_batches} remaining{C.RESET}")

        return remainder_loader, start_step
    except Exception as e:
        # Fallback to iterator-based skip
        if is_main:
            print(f"{C.YELLOW}Fast skip unavailable ({e}). Falling back to iterator skip.{C.RESET}")
        train_loader_iter = iter(train_loader)
        consumed = itertools.islice(train_loader_iter, start_step)
        skip_count = sum(1 for _ in consumed)
        if is_main and total_batches is not None:
            remaining_batches = total_batches - skip_count
            print(f"{C.CYAN}Skipped to position: {skip_count}; remaining {remaining_batches} batches{C.RESET}")
        return train_loader, skip_count
        # return enumerate(train_loader_iter, start=skip_count), skip_count

