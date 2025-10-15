import os
import subprocess
import sys
import torch
import json
import math
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, BertConfig
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
# from iter_data_loader import iter_data_loader
from data_loader import data_loader
import argparse
import time
import torch.distributed as dist
import traceback
from datetime import timedelta
# import torch.distributed as dist
from training_monitor import TrainingMonitor

import shutil

import torch.nn.functional as F
from training_utils import (
    log_first_batch_info, 
    forward_pass, 
    gradient_step, 
    update_progress_bars, 
    save_checkpoint,
    load_training_state,
    save_epoch_checkpoint,
    generate_special_tokens_map,
    safe_wait_for_everyone,
    save_once,
)
 
# ---- Lightweight JSONL logging (rank-0 only) ----
def _append_jsonl(path: str, record: dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Never let logging break training
        pass

# # Set CUDA_HOME to avoid DeepSpeed compilation issues
# import os
# os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.9')
# # Disable DeepSpeed compilation since we don't have nvcc
# os.environ.setdefault('DS_BUILD_OPS', '0')
# os.environ.setdefault('DS_SKIP_CUDA_CHECK', '1')

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

class EmptyDataset(Dataset):
    def __len__(self): 
        return 0
    def __getitem__(self, idx): 
        raise IndexError

# @torch.compile
def build_model(config, accelerator=None):
    config_path = config.get("model_config_path", None)
    gpt2_config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(gpt2_config)

    return model

import time
import datetime


def train_loop(
    accelerator,
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
    checkpoint_path,
    start_epoch,
    start_step=0,
):
    # Initialize training monitor with config values
    monitoring_config = config.get("monitoring", {})
    monitor = TrainingMonitor(
        window_size=monitoring_config.get("window_size", 20),
        spike_threshold=monitoring_config.get("spike_threshold", 0.3),
        oscillation_threshold=monitoring_config.get("oscillation_threshold", 0.15)
    )
    # train_loader = val_loader
    # save tokenizer
    # Save tokenizer manually since our custom tokenizer doesn't have save_pretrained
    os.makedirs(checkpoint_path, exist_ok=True)  # Ensure directory exists
    tokenizer_dest = os.path.join(checkpoint_path, "tokenizer.json")
    # tokenizer.save(tokenizer_dest)

    # tokenizer.save_pretrained(tokenizer_dest)
    
    # Auto-generate special_tokens_map.json for AutoTokenizer compatibility using refactored utility
    generate_special_tokens_map(tokenizer, checkpoint_path, accelerator, C)
    
    # Wait for main process to finish saving tokenizer files
    safe_wait_for_everyone(accelerator, "training_loop start")
    
    # Prepare local metrics log (rank-0 only, no barriers)
    is_main_global = accelerator.is_main_process
    logs_dir = os.path.join(checkpoint_path, "logs")
    if is_main_global:
        os.makedirs(logs_dir, exist_ok=True)
    metrics_path = os.path.join(logs_dir, "metrics.jsonl")
    # Use accelerator.print to avoid N prints across ranks
    accelerator.print(f"{C.BOLD}{C.CYAN}Starting training loop from epoch {start_epoch}, step {start_step} with config: {config}{C.RESET}")
    num_epochs = config["num_epochs"]
    if start_epoch >= num_epochs:
        accelerator.print(f"{C.GREEN}Training already completed. Exiting.{C.RESET}")
        return

    # 1) Compute number of steps and total tokens for ETA (initial estimate; bars recompute per-epoch)
    steps_per_epoch = len(train_loader)
    save_steps = max(1, steps_per_epoch // 2)
    print(f"{C.BLUE}Total steps per epoch: {steps_per_epoch}, save every {save_steps} steps{C.RESET}")
    if start_step > 0:
        print(f"{C.CYAN}Resuming from step {start_step} within epoch {start_epoch+1}{C.RESET}")
    # val_log_steps = max(1, steps_per_epoch // 200)
    # print(f"{C.BLUE}Validation every {val_log_steps} steps{C.RESET}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{C.BOLD}{C.MAGENTA}={'='*60}{C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}Starting epoch {epoch+1}/{num_epochs} (0-indexed: epoch {epoch}){C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}={'='*60}{C.RESET}")
        model.train()
        total_loss = 0.0
        processed_batches = 0
        epoch_start = time.time()
        log_steps = 1
        
        # Reset start_step after the first resumed epoch
        current_start_step = start_step if epoch == start_epoch else 0
        
        if epoch > start_epoch:
            print(f"{C.GREEN}✓ Successfully progressed to next epoch: {epoch+1}{C.RESET}")
        elif current_start_step > 0:
            print(f"{C.YELLOW}Resuming training in epoch {epoch+1} from step {current_start_step}{C.RESET}")
        else:
            print(f"{C.CYAN}Starting fresh epoch {epoch+1}{C.RESET}")
    
        # DEBUG
        # train_loader = val_loader

        # Only the main process draws the bars
        is_main = accelerator.is_main_process
        total_batches = len(train_loader)
        # Token‐based progress bar
        batch_bar = tqdm(
            total=total_batches,
            unit="batches",
            unit_scale=True,
            desc=f"Epoch {epoch+1}/{num_epochs} [Batches]",
            leave=True,  # Keep the bar visible after completion
            position=0,
            disable=not is_main,
        )
        # Loss bar (steps)
        loss_bar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1} [Loss]",
            leave=True,  # Keep the bar visible after completion
            position=1,
            bar_format="{l_bar}{bar}| {postfix}",
            disable=not is_main,
        )

        # Handle step resumption efficiently
        if epoch == start_epoch and current_start_step > 0:
            print(f"{C.YELLOW}Skipping {current_start_step} batches to resume from step {current_start_step}...{C.RESET}")
            print(f"{C.CYAN}Dataloader has {len(train_loader)} total batches{C.RESET}")
            
            # Validate that we can actually skip to this step
            if current_start_step >= len(train_loader):
                print(f"{C.RED}Error: Cannot skip to step {current_start_step}, dataloader only has {len(train_loader)} batches{C.RESET}")
                print(f"{C.YELLOW}This suggests the checkpoint is invalid or from a different dataset configuration{C.RESET}")
                # Reset to start from beginning of this epoch
                current_start_step = 0
                print(f"{C.CYAN}Falling back to starting from step 0 in epoch {epoch+1}{C.RESET}")
            
            if current_start_step > 0:  # Only skip if we still need to
                # Update progress bars to show skipped progress
                if is_main:
                    batch_bar.update(current_start_step)
                    loss_bar.update(current_start_step // log_steps)
                    loss_bar.set_postfix({"status": f"skipped to step {current_start_step}"})
                
                # Efficient iterator-based skipping using islice
                train_loader_iter = iter(train_loader)
                
                if is_main:
                    print(f"{C.CYAN}Skipping {current_start_step} batches using efficient islice...{C.RESET}")
                
                # Use islice to skip all batches at once - much faster than calling next() in a loop
                import itertools
                try:
                    # Consume (skip) the first current_start_step items without storing them
                    consumed = itertools.islice(train_loader_iter, current_start_step)
                    # Force consumption by converting to list and counting
                    skip_count = sum(1 for _ in consumed)
                    
                    if is_main:
                        print(f"{C.GREEN}Efficient skip completed: {skip_count} batches{C.RESET}")
                        
                except StopIteration:
                    if is_main:
                        print(f"{C.YELLOW}Iterator exhausted at skip_count={skip_count}. Using what we have.{C.RESET}")
                
                batch_iterator = enumerate(train_loader_iter, start=skip_count)
                
                # Add debug info about remaining batches (main process only)
                if is_main:
                    remaining_batches = len(train_loader) - skip_count
                    print(f"{C.CYAN}Skipped to position: {skip_count} (target was {current_start_step}){C.RESET}")
                    print(f"{C.CYAN}Will process {remaining_batches} remaining batches in this epoch{C.RESET}")
                    print(f"{C.GREEN}Skip completed, resuming training...{C.RESET}")
                
            else:
                # No skipping needed - start from beginning
                batch_iterator = enumerate(train_loader)
                if is_main:
                    print(f"{C.CYAN}Processing all {len(train_loader)} batches normally (no skipping){C.RESET}")
        else:
            # Not the resumption epoch, use normal iteration
            batch_iterator = enumerate(train_loader)
            
        # Main training loop - works for both resumed and normal cases
        batch_count = 0
        last_step = None
        for step, batch in batch_iterator:
            last_step = step
            batch_count += 1
            
            # Calculate the correct global step - simple for iterator-based skipping
            global_step = step
            
            # Debug logging for resumed training (main process only)
            if epoch == start_epoch and current_start_step > 0 and is_main:
                if step == current_start_step:  # First batch after skipping
                    print(f"{C.GREEN}Processing first resumed batch at step {step}{C.RESET}")
                elif batch_count == 10:
                    print(f"{C.BLUE}Training progressing normally, processed {batch_count} batches (step {step}){C.RESET}")
                elif batch_count % 500 == 0:
                    print(f"{C.BLUE}Progress check: processed {batch_count} batches, step {step}{C.RESET}")
            
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    # Forward pass
                    loss, logits = forward_pass(model, batch)
                    
                    # Log first batch details for debugging
                    if step == current_start_step and epoch == start_epoch:  # First resumed batch
                        log_first_batch_info(accelerator, batch, tokenizer, loss, C)
                    
                    # Handle non-finite loss
                    if not torch.isfinite(loss):
                        if is_main:
                            print(f"{C.YELLOW}Warning: non-finite loss at step {step} in epoch {epoch+1}. Replacing with 0.{C.RESET}")
                        loss = torch.zeros_like(loss)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient step (includes clipping, monitoring, optimizer step)
                loss_value = gradient_step(accelerator, model, optimizer, scheduler, monitor, config, step, loss, is_main, C)
                
                # Accumulate total loss
                total_loss += loss_value

                # Update progress bars and logging
                if step % log_steps == 0:
                    # Simple average loss calculation
                    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
                    
                    processed_batches += log_steps
                    update_progress_bars(batch_bar, loss_bar, monitor, loss_value, avg_loss, step, 
                                       log_steps, processed_batches, epoch_start, total_batches, is_main)
                    # Rank-0 JSONL logging for train loss
                    if is_main:
                        now_ts = time.time()
                        rec = {
                            "_type": "metric",
                            "time": now_ts,
                            "epoch": int(epoch),
                            "step": int(step),
                            "train/loss": float(loss_value),
                            "train/avg_loss": float(avg_loss),
                            # Provide dotted aliases for alternative plotting tools
                            "train.loss": float(loss_value),
                            "train.avg_loss": float(avg_loss),
                        }
                        _append_jsonl(metrics_path, rec)
                
                # Save checkpoint
                # save_checkpoint(accelerator, model, tokenizer, checkpoint_path, step, save_steps, is_main, C, epoch)
        
        batch_bar.close()
        loss_bar.close()

        print(f"\n{C.GREEN}✓ Completed epoch {epoch+1}/{num_epochs}{C.RESET}")
        print(f"{C.CYAN}Total batches processed: {total_batches}{C.RESET}")
        
        # End‑of‑epoch checkpoint (main-only, with barriers)
        # accelerator.wait_for_everyone()
        # if accelerator.is_main_process:
        save_epoch_checkpoint(accelerator, model, tokenizer, checkpoint_path, epoch, is_main, C)
        
        print(f"{C.YELLOW}Epoch {epoch+1} checkpoint saved, ready to progress to epoch {epoch+2}{C.RESET}")
        print(f"{C.DIM}Training state will be saved as: epoch={epoch+1}, step=0{C.RESET}")
        
        # Run validation only on main process with unprepared loader (avoid barriers)
        model.eval()
        
        with torch.no_grad():
            seen = 0
            val_total = 0.0
            val_weight = 0
            total_masked = 0
            for val_batch in tqdm(val_loader, desc="Validating (full)", disable=not is_main):
                val_loss, logits = forward_pass(model, val_batch)
                val_loss = accelerator.gather(val_loss).mean()
                # Weight by number of masked tokens when available to reduce variance
                try:
                    labels = val_batch.get("labels") if isinstance(val_batch, dict) else None
                    labels = accelerator.gather(labels) if labels is not None else None
                    if labels is not None:
                        masked = (labels != -100)
                        accelerator.gather(masked)
                        weight = int(masked.sum().item())
                    else:
                        weight = 0
                except Exception:
                    weight = 0
                if weight <= 0:
                    weight = 1
                val_total += float(val_loss) * weight
                val_weight += weight
                total_masked += max(0, weight)
                seen += 1
            avg_val_loss = val_total / max(1, val_weight)
            print(f"{C.CYAN}[End-epoch] epoch {epoch+1}: avg_val_loss={avg_val_loss:.4f} over {seen} batches (masked tok={total_masked}){C.RESET}")
            # Log validation loss and epoch-end marker (rank-0 only)
            now_ts = time.time()
            end_step = int(last_step) if last_step is not None else int(steps_per_epoch - 1)
            rec_val = {
                "_type": "metric",
                "time": now_ts,
                "epoch": int(epoch + 1),
                "step": end_step,
                "val/loss": float(avg_val_loss),
                "val.loss": float(avg_val_loss),
                "val.batches": int(seen),
                "val.masked_tokens": int(total_masked),
            }
            if is_main:
                _append_jsonl(metrics_path, rec_val)
                rec_marker = {
                    "_type": "epoch_end",
                    "time": now_ts,
                    "epoch": int(epoch + 1),
                    "step": end_step,
                }
                _append_jsonl(metrics_path, rec_marker)
                
        # Print epoch monitoring summary
        if is_main:
            epoch_monitor_stats = monitor.get_stats()
            print(f"{C.BOLD}{C.GREEN}[Epoch {epoch+1} Summary] {epoch_monitor_stats}{C.RESET}")
            if monitor.spike_count > 0 or monitor.oscillation_count > 0 or monitor.explosion_count > 0:
                print(f"{C.YELLOW}⚠️  Training stability issues detected this epoch. Consider reducing learning rate if issues persist.{C.RESET}")
                
        model.train()
    
        # Wait for main process to finish validation before continuing
        # accelerator.wait_for_everyone()
        # accelerator.wait_for_everyone()


def main():
    print(torch._dynamo.list_backends())

    # Fast kernels (safe on Ampere+; no effect otherwise)
    if bool(os.environ.get("ALLOW_TF32", "1") == "1"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            from torch.nn.attention import sdp_kernel
            # Prefer flash if available, fallback to mem_efficient
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass

    # 2) pin this process to the correct CUDA device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # grab LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT from torchrun/accelerate launch
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    # DDP speed-ups
    ddp_kwargs = DistributedDataParallelKwargs(
        static_graph=True,                 # fixed graph speeds up allreduce
        gradient_as_bucket_view=True,      # lower memory + faster
        find_unused_parameters=False,
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config["grad_accum"],
        dataloader_config=DataLoaderConfiguration(even_batches=True, split_batches=False),
        # kwargs_handlers=[ddp_kwargs],
    )

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint")
    # Enforce tokenizer model_max_length from config (fallback to max_position_embeddings, then 512)
    tkn_max_len = int(config.get("model_max_length", config.get("max_position_embeddings", 512)))
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_path", "./model_babylm_gpt2_16k"),
        model_max_length=tkn_max_len,
    )
    # Explicitly set (some fast tokenizers ignore constructor kwarg)
    try:
        tokenizer.model_max_length = tkn_max_len
    except Exception:
        pass
    print(f"{C.CYAN}Tokenizer model_max_length set to {getattr(tokenizer, 'model_max_length', 'N/A')}{C.RESET}")

    print(f"{C.CYAN}Creating data loaders...{C.RESET}")
    train_loader, val_loader, test_loader, collate_fn, total_tokens_train = data_loader(config, tokenizer, config["cache_path"])
    print(f"{C.GREEN}Data loaders created successfully{C.RESET}")
    # Build the GPT-2 model from scratch based on our config
    model = build_model(config)
    model.resize_token_embeddings(len(tokenizer))
    
    # # Enable gradient checkpointing for memory efficiency but comes at a speed penalty of 15-20%s
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_enable()
    #     print(f"{C.BLUE}Enabled gradient checkpointing for memory efficiency{C.RESET}")

    # if accelerator.is_main_process:
    #     print(f"{C.BLUE}LR base={base_lr:.2e} scaled={scaled_lr:.2e} (effective_bs={effective_bs}){C.RESET}")

    # Load the saved epoch and step from training state EARLY to align scheduler on restarts
    start_epoch, start_step = load_training_state(checkpoint_path)
    print(f"\n{C.BOLD}{C.BLUE}=== TRAINING STATE (pre-opt/sched) ==={C.RESET}")
    print(f"{C.BLUE}Loaded training state: epoch={start_epoch}, step={start_step}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}======================================{C.RESET}\n")

    # --- Effective batch & conservative late-stage scaling ---
    base_lr    = float(config.get("learning_rate", 1e-4))    # sensible default for BERT-base-ish late training
    grad_accum = int(config.get("grad_accum", 1))
    world_size = getattr(accelerator, "num_processes", int(os.environ.get("WORLD_SIZE", 1)))
    effective_bs = int(config["batch_size"]) * max(1, grad_accum) * max(1, world_size)

    # Sub-linear scaling (AdamW-friendly). Disable by setting alpha=0.0
    alpha = float(config.get("lr_scale_alpha", 0.5))  # 0.5–0.7 work well; or 0.0 to disable scaling
    scale = (effective_bs / 256.0) ** alpha
    peak_cap = float(config.get("peak_lr_cap", 1e-4))  # late-stage cap; avoids going to 3e-4
    scaled_lr = max(1e-5, min(base_lr * scale, peak_cap))

    if accelerator.is_main_process:
        print(f"{C.BLUE}LR base={base_lr:.2e} scaled={scaled_lr:.2e} "
            f"(effective_bs={effective_bs}, alpha={alpha}){C.RESET}")

    # --- Build AdamW param groups (no decay for LN & bias; optional: embeddings) ---
    no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keys):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    weight_decay = float(config.get("weight_decay", 0.01))
    # Use a *separate* optimizer eps; don't reuse layer_norm_eps
    optim_eps = float(config.get("optimizer_eps", 1e-8))
    betas = (0.9, 0.95)

    # --- Optimizer selection (prefer fused AdamW for speed) ---
    use_fused = bool(config.get("use_fused_adamw", True))
    use_foreach = True
    adamw_cls = torch.optim.AdamW
    param_groups = [
        {"params": decay_params,   "weight_decay": weight_decay, "lr": scaled_lr},
        {"params": nodecay_params, "weight_decay": 0.0,          "lr": scaled_lr},
    ]
    optimizer = adamw_cls(param_groups, lr=scaled_lr, betas=betas, eps=optim_eps,
                        fused=use_fused if "fused" in adamw_cls.__init__.__code__.co_varnames else False)

    # --- Cosine schedule with warmup as a fraction of *optimizer updates* ---
    # Make sure you compute these with the *true* number of batches and grad_accum
    num_train_batches = len(train_loader)
    updates_per_epoch = math.ceil(num_train_batches / max(1, grad_accum))
    # Derive resume progress in optimizer-update units from saved state
    resume_updates = (start_epoch * updates_per_epoch) + (start_step // max(1, grad_accum))
    total_updates = int(config.get("num_epochs", 30)) * updates_per_epoch

    warmup_prop = float(config.get("warmup_steps_proportion", 0.06))
    warmup_updates = max(1, int(warmup_prop * total_updates))

    min_lr = float(config.get("min_lr", 1e-5))
    floor_ratio = min_lr / float(scaled_lr)

    def lr_lambda(step):
        # step is the *optimizer* update index (after grad_accum)
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        t = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        # cosine from 1.0 -> floor_ratio
        return floor_ratio + 0.5 * (1.0 - floor_ratio) * (1.0 + math.cos(math.pi * t))

    # --- Gradient clipping: engage earlier to stop "near explosion" warnings ---
    max_grad_norm = float(config.get("max_grad_norm", 1.0))  # ↓ from 1.5
    
    if config.get("optimizer", "adamw").lower() == "lamb":
        print(f"{C.BLUE}Using LAMB optimizer with robust param grouping{C.RESET}")

        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight", "embedding", "word_embedding")
        head_modules = ("classifier.", "lm_head.")

        decay, nodecay, head = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in no_decay_keys):
                nodecay.append(p)
            elif any(h in n for h in head_modules):
                head.append(p)
            else:
                decay.append(p)

        wd = float(config.get("weight_decay", 0.01))
        head_wd = float(config.get("head_weight_decay", 0.1))
        optimizer_grouped_parameters = [
            {"params": decay,   "weight_decay": wd},
            {"params": head,    "weight_decay": head_wd},
            {"params": nodecay, "weight_decay": 0.0},
        ]

        if accelerator.is_main_process:
            print(f"{C.DIM}LAMB param groups — decay: {sum(p.numel() for p in decay):,} params, head: {sum(p.numel() for p in head):,}, no-decay: {sum(p.numel() for p in nodecay):,}{C.RESET}")

        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=scaled_lr,  # keep same scaled LR logic and caps as AdamW path
            betas=(0.9, 0.98),
            eps=float(config.get("optimizer_eps", 1e-6)),
        )
   
    # Prepare model, optimizer, and train/test loaders. Leave val_loader unprepared for main-only eval.
    model, optimizer, train_loader, test_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader, val_loader
    )
    # Keep val_loader unprepared to avoid NCCL timeouts during validation

    if accelerator.is_main_process:
        # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # Print model architecture, parameter counts, and full size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_mb = total_bytes / (1024 ** 2)
        print(f"{C.BOLD}{C.CYAN}===== Model Summary ====={C.RESET}", flush=True)
        # Colorize the full multi-line repr of the model
        print(f"{C.DIM}{C.RED}{model}{C.RESET}", flush=True)
        print(f"{C.CYAN}Total parameters:     {total_params:,}{C.RESET}", flush=True)
        print(f"{C.CYAN}Trainable parameters: {trainable_params:,}{C.RESET}", flush=True)
        print(f"{C.CYAN}Approx. model size:   {total_mb:.2f} MB{C.RESET}", flush=True)
        print(f"{C.BOLD}{C.CYAN}=========================={C.RESET}", flush=True)
        # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Build LR scheduler attached to the prepared optimizer.
    # Use last_epoch=-1 to seed 'initial_lr' in param groups; then optionally jump to resume position.
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # Decide whether to reinitialize scheduler state on restart
    reinit_scheduler = bool(config.get("reinitialize_scheduler", True))

    # If we want to reinitialize the scheduler, delay registering it until after load_state
    if not reinit_scheduler:
        # Ensure the scheduler is part of the checkpoint state
        accelerator.register_for_checkpointing(scheduler)

    if checkpoint_path and os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
        try:
            accelerator.print(f"{C.CYAN}Loading checkpoint from {checkpoint_path}{C.RESET}")
            accelerator.load_state(checkpoint_path)
            accelerator.print(f"{C.GREEN}Checkpoint load completed.{C.RESET}")
        except Exception as e:
            accelerator.print(
                f"{C.YELLOW}Warning: could not fully load accelerator state from {checkpoint_path}: {e}. "
                f"Will continue; optimizer/scheduler may start fresh.{C.RESET}"
            )
            # Fallback: try to restore model weights only if saved via save_pretrained
            try:
                unwrapped = accelerator.unwrap_model(model)
                weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                if os.path.isfile(weights_path):
                    state_dict = torch.load(weights_path, map_location="cpu")
                    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
                    accelerator.print(
                        f"{C.YELLOW}Loaded model weights only (missing={len(missing)}, unexpected={len(unexpected)}).{C.RESET}"
                    )
            except Exception as ee:
                accelerator.print(f"{C.RED}Model-only weight restore failed: {ee}{C.RESET}")
    else:
        accelerator.print(f"{C.YELLOW}No checkpoint found at {checkpoint_path}, starting from scratch.{C.RESET}")

    # After potential state load, if we reinitialized the scheduler, align and register it now
    if reinit_scheduler:
        # Advance scheduler to the resume position in optimizer-update units
        if resume_updates > 0:
            scheduler.step(resume_updates)
        accelerator.register_for_checkpointing(scheduler)
    
    print(f"\n{C.BOLD}{C.BLUE}=== TRAINING STATE DEBUG ==={C.RESET}")
    print(f"{C.BLUE}Loaded training state: epoch={start_epoch}, step={start_step}{C.RESET}")
    print(f"{C.BLUE}Total epochs configured: {config['num_epochs']}{C.RESET}")
    print(f"{C.BLUE}Will run epochs: {start_epoch} → {config['num_epochs']-1} (range({start_epoch}, {config['num_epochs']}))){C.RESET}")
    
    # Interpret the training state correctly:
    # - If step > 0: we're in the middle of start_epoch, resume from start_step
    # - If step = 0: we completed start_epoch-1, start fresh at start_epoch
    if start_step > 0:
        accelerator.print(f"{C.GREEN}Resuming training in the middle of epoch {start_epoch + 1} at step {start_step}{C.RESET}")
    elif start_epoch > 0:
        accelerator.print(f"{C.GREEN}Resuming training at the beginning of epoch {start_epoch + 1} (completed {start_epoch} epochs){C.RESET}")
    else:
        accelerator.print(f"{C.CYAN}Starting training from the beginning{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}==========================={C.RESET}\n")
    
    # train_loader = test_loader
    save_once(accelerator, tokenizer, checkpoint_path)
    train_loop(accelerator, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, checkpoint_path, start_epoch, start_step)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise