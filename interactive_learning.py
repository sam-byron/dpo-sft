# bnc_interactive_train.py
# ==================================================================================
# PERFORMANCE OPTIMIZATIONS APPLIED:
# ==================================================================================
# Inference Optimizations:
#   - torch.compile with max-autotune for all inference-only models (reference, critic, caregiver)
#   - torch.inference_mode() for generation and uncertainty calculation (faster than no_grad)
#   - Mixed precision (bfloat16) for all forward passes
#   - Flash Attention 2 support (if available)
#   - KV-cache enabled (use_cache=True) for autoregressive generation
#   
# Training Optimizations:
#   - Fused AdamW optimizer (faster parameter updates on CUDA)
#   - Gradient checkpointing (memory efficiency)
#   - TF32 matmul enabled for faster computation on Ampere+ GPUs
#   - cudnn.benchmark for optimal kernel selection
#   - Gradient zeroing with set_to_none=True (memory efficiency)
#   - High precision matmul (float32_matmul_precision="high")
#
# Data Pipeline Optimizations:
#   - Fast tokenizer (use_fast=True)
#   - Left padding for efficient batch generation
#   - Parallel data loading (num_workers=4, pin_memory=True)
#   - Batch processing for caregiver corrections
#
# Compiler Configuration:
#   - Dynamic shapes disabled for stable inference graphs
#   - Cache size limit increased to 64 (default 8) for text generation workloads
#   - CUDA graphs disabled to avoid dynamic shape warnings
#   - Suppress recompilation warnings
# ==================================================================================
import os, re, math, random, json, argparse, itertools, time
import pickle  # Add this import
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling, AutoConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, PeftModel
from load_lora import load_and_merge_lora, load_adapter_config

# Import the color logger
from color_logger import get_logger, MLColors, Fore, Style
# BLiMP evaluation utilities
from blimp import run_subset, ensure_subsets_list, pick_split

from interactive_utils import Caregiver, CaregiverOutput, build_contrastive_pairs, dpo_loss, eval_blimp_hf, get_uncertainty, generate, ce_targets, kl_to_ref, logprob_sum, mini_morph, WordBudget, combined_loss

from prepare_data import load_or_prepare_dataset

# Disable torch compile for debugging (single line toggle)
torch._dynamo.config.disable = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)
torch.manual_seed(7)

# Disable torch.compile for debugging (single line toggle)
torch._dynamo.config.disable = True 

# Add after imports & seeds (near top, after DEVICE):
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Initialize the color logger
logger = get_logger("BabyLM-Interactive")

# # Make compile robust to dynamic shapes and fall back instead of crashing
try:
    import torch._dynamo as dynamo
    import torch._inductor.config as inductor_config
    
    # Configure dynamo for better dynamic shape handling
    dynamo.config.dynamic_shapes = True
    dynamo.config.suppress_errors = True
    dynamo.config.verbose = False  # Reduce verbosity
    
    # Increase recompile limit for dynamic text generation workloads
    dynamo.config.cache_size_limit = 256  # Increased from 64 to handle more variations
    
    # Configure CUDAGraph settings - be more permissive
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None  # Silence warnings
    
    # Additional configs to improve compilation success
    inductor_config.max_autotune = False  # Disable to avoid Triton shared memory errors
    inductor_config.triton.cudagraphs = False  # Disable cudagraphs entirely (they're causing the warnings)
    
    logger.info("Configured torch.compile settings for dynamic shapes (cache_size_limit=256, max_autotune=False)")
except Exception as e:
    logger.warning(f"torch._dynamo/inductor config not available: {e}")

from datasets import load_dataset, get_dataset_config_names

def build_model(checkpoint_path):

    config_path = os.path.join(checkpoint_path, "config.json")
    model_config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(model_config)

    return model

# -----------------------
# Main training
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bnc_name", type=str, default="deven367/babylm-100M-bnc-spoken", help="HF dataset name if available (e.g., 'bnc' or your org/dataset')")
    ap.add_argument("--bnc_dir", type=str, default="", help="Path to BNC .txt files if not using HF")
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--model_path", type=str, default=None, help="Optional path to a LoRA adapter directory (with adapter_config.json) to load and merge into base using load_lora.py. If provided, overrides --model_name.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--use_dpo", action="store_true")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_dir", type=str, default="./ckpts_bnc_interactive")
    ap.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = ap.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    checkpoint_path = os.path.join(config["cache_path"], "checkpoint")
    model_config_path = os.path.join(checkpoint_path, "config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    # Defaults for initial eval placeholders to satisfy static analyzers
    bl0 = 0.0
    mo0 = 0.0

    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info(f"🚀 Starting interactive learning with {Fore.YELLOW}{args.model_name}{Style.RESET_ALL}")
    logger.info(f"Device: {Fore.GREEN}{DEVICE}{Style.RESET_ALL}")
    logger.info(f"Batch size: {args.batch_size}, Steps: {args.steps:,}, LR: {args.lr}")
    logger.info(f"Save directory: {args.save_dir}")
    
    # Performance optimization notices
    if args.batch_size < 16:
        logger.warning(f"⚠️  Small batch size ({args.batch_size}) may cause slow training. Consider increasing to 32-64 for better GPU utilization.")
    logger.info(f"🚀 Performance optimizations: batch generation, parallel data loading, profiling enabled")

    # Model/tokenizer loading
    
    if args.model_path:
        logger.info(f"Loading base + LoRA (adapters trainable only) from: {Fore.CYAN}{args.model_path}{Style.RESET_ALL}")
        # Discover base model from adapter config, else fall back to --model_name
        try:
            student = build_model(checkpoint_path)
            weights_path = os.path.join(args.model_path, "pytorch_model.bin")
            if os.path.isfile(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                missing, unexpected = student.load_state_dict(state_dict, strict=False)
                student.to(DEVICE)
                logger.info(
                    f"Loaded model weights only (missing={len(missing)}, unexpected={len(unexpected)})."
                )
        except Exception as ee:
            logger.error(f"Model-only weight restore failed: {ee}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.model_max_length = model_config.get("n_ctx")
        print(f"Tokenizer model_max_length set to {getattr(tokenizer, 'model_max_length', 'N/A')}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder models need left padding
        tokenizer.padding_side = 'left'
        student.train()
        student.to(DEVICE)
        logger.info("Loaded student model")
    else:
        logger.info("Loading tokenizer...")
        # Use fast tokenizer explicitly for better performance
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Fix padding side for decoder-only models (critical for generation quality)
        tokenizer.padding_side = 'left'  # This should be the ONLY place we set padding_side
        logger.info(f"Set tokenizer padding_side to: {Fore.GREEN}left{Style.RESET_ALL} (required for decoder-only models)")
        logger.info(f"Using fast tokenizer: {Fore.GREEN}{tokenizer.is_fast}{Style.RESET_ALL}")
        logger.info(f"Suppressed tokenizer parallelism warnings via environment variable")

        # Student model with optimizations
        logger.info("Setting up student model...")
        model_config = AutoConfig.from_pretrained(args.model_name)
        
        # Try to enable Flash Attention 2 if available
        try:
            model_config._attn_implementation = "flash_attention_2"
            logger.info("Attempting to use Flash Attention 2 for faster inference")
        except Exception:
            pass
        
        student = AutoModelForCausalLM.from_config(model_config)
        student.to(DEVICE)
        student.train()
        
        # Check if Flash Attention was successfully enabled
        if hasattr(student.config, '_attn_implementation'):
            logger.success(f"✓ Attention implementation: {student.config._attn_implementation}")
        else:
            logger.info("Using default attention implementation")
    
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Trainable parameters: {Fore.GREEN}{trainable_params:,}{Style.RESET_ALL} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # # Enable gradient checkpointing for memory efficiency (if model supports it)
    # if hasattr(student, 'gradient_checkpointing_enable'):
    #     try:
    #         student.gradient_checkpointing_enable()
    #         logger.success("✓ Enabled gradient checkpointing for memory efficiency")
    #     except Exception as e:
    #         logger.warning(f"Gradient checkpointing not available: {e}")
    
    import copy
    from datasets import load_from_disk
    
    # Reference (frozen) - aggressively optimize for inference
    logger.info("Loading reference model...")
    reference = copy.deepcopy(student)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False
    reference.to(DEVICE)
    
    # Compile reference with reduce-overhead (faster compilation, still good performance)
    try:
        reference = torch.compile(reference, mode="reduce-overhead", fullgraph=False, dynamic=False)
        logger.success("✓ Compiled reference model with reduce-overhead")
    except Exception as e:
        logger.warning(f"torch.compile (reference) skipped: {e}")

    logger.info("Setting up caregiver...")
    caregiver = Caregiver()
    
    # Compile caregiver model for faster inference (wrap, don't reassign)
    original_model = caregiver.model
    try:
        compiled_model = torch.compile(original_model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        # Replace the model through __dict__ to bypass type checking
        caregiver.__dict__['model'] = compiled_model
        logger.success("✓ Compiled caregiver model with reduce-overhead")
    except Exception as e:
        logger.warning(f"torch.compile (caregiver) skipped: {e}")

    # Data stream
    bnc_name = args.bnc_name if args.bnc_name else None
    bnc_dir  = args.bnc_dir if args.bnc_dir else None
    logger.info(f"Setting up data stream from: {bnc_name or bnc_dir or 'default'}")
    
    # Load or prepare dataset (simplified)
    ds = load_or_prepare_dataset(config, logger)
    
    def identity_collate_fn(batch):
        """
        Identity collate function - returns batch as-is without any processing.
        Just passes through whatever the dataset returns.
        """
        return batch
    
    def sentence_batch_collate_fn(batch):
        """
        Collate function that flattens all sentences from all dataset items
        into individual batch elements. Each sentence becomes one training example.
        
        Input: batch = [list1, list2, list3, list4] where each list contains sentences
        Output: [sent1, sent2, sent3, sent4, sent5, sent6, ...] (flattened)
        
        Example:
        - Item 0: ["Sentence A", "Sentence B", "Sentence C"] 
        - Item 1: ["Sentence D", "Sentence E"]
        - Item 2: ["Sentence F"]
        - Item 3: ["Sentence G", "Sentence H"]
        Result: ["Sentence A", "Sentence B", "Sentence C", "Sentence D", "Sentence E", "Sentence F", "Sentence G", "Sentence H"]
        """
        all_sentences = []
        
        for item in batch:
            if isinstance(item, list):
                # Each item is a list of sentences - add all to batch
                all_sentences.extend(item)
            elif isinstance(item, str):
                # Single string - add directly
                all_sentences.append(item)
            else:
                # Fallback - convert to string
                all_sentences.append(str(item))
        
        # Filter out empty sentences
        all_sentences = [s.strip() for s in all_sentences if s.strip()]
        
        return all_sentences
    # Reduce workers to avoid tokenizer parallelism issues with streaming datasets
    # Add drop_last=True to stabilize batch shape for compiled graphs
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True, collate_fn=sentence_batch_collate_fn)

    # Optim & sched
    logger.info("Setting up optimizer and scheduler...")
    # Try to use fused AdamW for better performance (requires CUDA)
    try:
        opt = torch.optim.AdamW(
            student.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.95), 
            weight_decay=0.01,
            fused=True  # Fused kernel for faster updates
        )
        logger.success("✓ Using fused AdamW optimizer")
    except Exception as e:
        logger.warning(f"Fused AdamW not available, using standard: {e}")
        opt = torch.optim.AdamW(
            student.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.95), 
            weight_decay=0.01
        )
    
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=500, num_training_steps=args.steps)

    budget = WordBudget(20_000_000)
    logger.info(f"Word budget: {Fore.YELLOW}{budget.limit:,}{Style.RESET_ALL} words")

    logger.info("Running initial evaluation on REAL BLiMP…")
    # bl0, per_cat0 = eval_blimp_hf(student, tokenizer, n_per_cat=100, max_len=64, progress=True)
    # mo0 = mini_morph(student, tokenizer)
    # logger.eval(0, f"Initial BLiMP (real): {bl0:.3f}")

    
    logger.success("🎯 Starting training loop...")
    logger.info(f"📊 Progress will be tracked with {'tqdm' if hasattr(logger, 'progress_bar') else 'basic'} progress bars")
    
    # Create progress bar for training
    progress_bar = logger.create_progress_bar(args.steps, "🚀 Training")

    step = 0
    step_times = {"data": 0.0, "generate": 0.0, "correct": 0.0, "forward": 0.0, "backward": 0.0}
    # Running average (EMA) for loss displayed in tqdm
    avg_loss = None
    avg_decay = 0.98  # higher = smoother
    
    # Caregiver audit: log corrections every 25 steps
    audit_path = os.path.join(args.save_dir, "caregiver_audit.json")
    critic_name = 'grammarly/coedit-large'
    # Load critic (prefer causal LM; fallback to seq2seq if needed)
    try:
        critic = AutoModelForCausalLM.from_pretrained(
            critic_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception:
        from transformers import AutoModelForSeq2SeqLM
        critic = AutoModelForSeq2SeqLM.from_pretrained(
            critic_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False
    
    # Compile critic for faster inference (reduce-overhead mode for faster compilation)
    try:
        critic = torch.compile(critic, mode="reduce-overhead", fullgraph=False, dynamic=False)
        logger.success("✓ Compiled critic model with reduce-overhead")
    except Exception as e:
        logger.warning(f"torch.compile (critic) skipped: {e}")

    dpo_interval = 5
    dpo_warmup = 250  # steps
    last_L_dpo = torch.tensor(0.0, device=DEVICE)  # for smoother logging

    for batch in loader:
        step_start = time.time()
        step += 1
        
        prefixes = batch  # Already a list of strings due to identity_collate_fn

        # # Light nudge: append a small cue token to expose the locus (helps heuristics)
        # prefixes = [p + " is" if re.search(r"\b\w+s\b$", p) else p for p in prefixes]

        # Generate short student attempts
        gen_start = time.time()
        with torch.inference_mode():  # Faster than no_grad for pure inference
            attempts = generate(student, tokenizer, prefixes, temperature=0.9, top_p=0.9)
        
        # Ensure strings for downstream typing (guard lints)
        # attempts = [a if isinstance(a, str) else "" for a in attempts]
        gen_time = time.time() - gen_start
        step_times["generate"] += gen_time
        
        # NEW: Calculate uncertainty and select samples for correction
        correct_start = time.time()
        with torch.inference_mode():  # Use inference_mode for uncertainty calculation
            uncertainties = get_uncertainty(student, tokenizer, prefixes, attempts)

        # Correct top 20% most uncertain samples
        threshold = sorted(uncertainties, reverse=True)[int(0.2 * len(uncertainties))]
        needs_correction = [u >= threshold for u in uncertainties]
        n_correct = sum(needs_correction)

        # Only invoke caregiver for flagged samples
        if n_correct > 0:
            # Extract samples needing correction
            idxs_to_correct = [i for i, flag in enumerate(needs_correction) if flag]
            prefixes_subset = [prefixes[i] for i in idxs_to_correct]
            attempts_subset = [attempts[i] if isinstance(attempts[i], str) else "" for i in idxs_to_correct]
           
            # Invoke caregiver only on subset
            try:
                outs_subset = caregiver.correct_batch(prefixes_subset, attempts_subset)
                
                # Audit caregiver corrections every 25 steps
                if step % 25 == 0:
                    audit_records = []
                    for idx, o_sub in zip(idxs_to_correct, outs_subset):
                        # lines = o_sub.corrected.split('\n')
                        # flatten lines into single string with | separator
                        # lines = re.split(r'[\n|]+', o_sub.corrected)
                        # text = " | ".join([line.strip() for line in o_sub if line.strip()])
                        audit_records.append({
                            "step": int(step),
                            "prefix": str(prefixes[idx]),
                            "student": str(attempts[idx]),
                            "caregiver_corrected": o_sub.corrected,
                        })
                    
                    # Append to audit file
                    with open(audit_path, "a", encoding="utf-8") as f:
                        for rec in audit_records:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                # Merge back: corrected samples + unchanged samples
                outs = []
                correct_iter = iter(outs_subset)
                for i, flag in enumerate(needs_correction):
                    if flag:
                        outs.append(next(correct_iter))
                    else:
                        # Keep student's original output
                        outs.append(CaregiverOutput(attempts[i]))
            except Exception as e:
                logger.warning(f"⚠️  Caregiver batch failed: {e}")
                outs = [CaregiverOutput(str(att or "")) for att in attempts]
        else:
            # No samples need correction
            outs = [CaregiverOutput(str(att or "")) for att in attempts]
        
        y_star = [o.corrected for o in outs]
        # y_star = text
        correct_time = time.time() - correct_start
        step_times["correct"] += correct_time

        # # Track correction statistics for logging
        # if step % 200 == 0:  # Less frequent than main logging
        #     tag_counts = {}
        #     for o in outs:
        #         tag_counts[o.tag] = tag_counts.get(o.tag, 0) + 1
            
        #     if any(tag != "other" for tag in tag_counts):
        #         corrections_msg = []
        #         for tag, count in sorted(tag_counts.items()):
        #             if tag != "other" and count > 0:
        #                 color = Fore.GREEN if tag in ["agreement", "reflexive"] else Fore.CYAN
        #                 corrections_msg.append(f"{color}{tag}: {count}{Style.RESET_ALL}")
                
        #         if corrections_msg:
        #             logger.debug(f"📝 Corrections in batch: {', '.join(corrections_msg)}")

        # --- Compute core losses first ---
        forward_start = time.time()
        L_sft = ce_targets(student, tokenizer, prefixes, y_star)
        L_kl  = kl_to_ref(student, reference, tokenizer, prefixes, attempts)  # ✅ Use attempts, not y_star
        L_dpo = torch.tensor(0.0, device=DEVICE)  # default (in case we skip or no pairs)
        # do_dpo = args.use_dpo and (step % dpo_interval == 0)
        do_dpo = True

        if do_dpo:
            xs, chosen, rejected = build_contrastive_pairs(
                student, reference, tokenizer, prefixes_subset, attempts_subset,
                k_per_prefix=6, margin=0.4, max_len=15, critic=critic
            )
            if xs and chosen and rejected:
                L_dpo = dpo_loss(student, reference, tokenizer, xs, chosen, rejected, beta=0.5)
                last_L_dpo = L_dpo.detach()
        else:
            # Use last computed DPO loss for smoother loss
            L_dpo = last_L_dpo.detach()

        # Amortize and ramp DPO weight to avoid spikes
        base_dpo_weight = (0.5 / dpo_interval) if args.use_dpo else 0.0
        ramp = min(1.0, step / dpo_warmup) if args.use_dpo else 0.0
        dpo_weight = base_dpo_weight * ramp

        # --- Combine losses (final total) ---
        loss = L_sft + 0.75 * L_kl + dpo_weight * L_dpo
        step_times["forward"] += time.time() - forward_start

        # Update running average (EMA)
        loss_val = float(loss.item())
        avg_loss = loss_val if avg_loss is None else (avg_decay * avg_loss + (1.0 - avg_decay) * loss_val)

        backward_start = time.time()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step()
        step_times["backward"] += time.time() - backward_start

        # Budget: only count texts we train ON (y*)
        budget.add(y_star)
        if not budget.ok():
            logger.warning("⚠️  Reached 100M word limit. Stopping training.")
            break

        # Update progress bar with current metrics (ensure all losses are shown)
        current_lr = sched.get_last_lr()[0]
        
        # Use dictionary method for progress bar updates (order matters in tqdm)
        # Show running average first; remove Corr from the bar
        progress_metrics = {
            "Avg": f"{avg_loss:.3f}",
            "Loss": f"{loss_val:.3f}",
            "SFT": f"{L_sft.item():.3f}",
            "KL": f"{L_kl.item():.3f}",
            "DPO": f"{(dpo_weight * (L_dpo if do_dpo else last_L_dpo)).item():.3f}",
            "LR": f"{current_lr:.1e}",
            "Words": f"{budget.used/1e6:.1f}M",
        }
        logger.update_progress(step, **progress_metrics)

        # Detailed logging every 100 steps (less frequent than progress bar)
        if step % 100 == 0:
            # Use the enhanced loss display from the logger
            losses = {"Total": loss.item(), "SFT": L_sft.item(), "KL": L_kl.item(), "DPO": L_dpo.item()}
            thresholds = {"Total": (2.5, 4.0), "SFT": (2.0, 3.0), "KL": (0.1, 0.5), "DPO": (1.0, 2.0)}
            
            loss_display = logger.loss_display(losses, thresholds)
            
            # Calculate correction rate for display
            correction_rate = n_correct / len(prefixes)
            corr_color = Fore.GREEN if correction_rate < 0.3 else Fore.YELLOW if correction_rate < 0.5 else Fore.RED
            
            extra_info = (
                f"LR: {Fore.CYAN}{current_lr:.2e}{Style.RESET_ALL} | "
                f"📚 {Fore.CYAN}{budget.used/1e6:.1f}M{Style.RESET_ALL} words | "
                f"🎯 Corr: {corr_color}{correction_rate:.1%}{Style.RESET_ALL} ({n_correct}/{len(prefixes)})"
            )
            
            # Performance breakdown
            total_time = sum(step_times.values())
            if total_time > 0:
                perf_info = []
                for component, t in step_times.items():
                    pct = (t / total_time) * 100
                    color = Fore.RED if pct > 40 else Fore.YELLOW if pct > 20 else Fore.GREEN
                    perf_info.append(f"{component}: {color}{pct:.1f}%{Style.RESET_ALL}")
                
                logger.step(step, f"📊 {loss_display} | {extra_info}")
                logger.debug(f"⏱️  Performance: {' | '.join(perf_info)} (avg: {total_time/step:.2f}s/step)")
            else:
                logger.step(step, f"📊 {loss_display} | {extra_info}")

        # Evaluation and checkpointing
        if step % args.eval_every == 0:
            logger.info("🔍 Running BLiMP (real) evaluation…")
            bl, per_cat = eval_blimp_hf(student, tokenizer, n_per_cat=100, max_len=64, progress=True)
            logger.eval(step, f"BLiMP (real): {bl:.3f}")

            
            # Update progress bar with evaluation metrics
            eval_metrics = {
                "SFT_loss": L_sft.item(),
                "KL_loss": L_kl.item(), 
                "DPO_loss": L_dpo.item(),
                "BLiMP_score": bl,
                "Morph_score": mo0 if 'mo0' in locals() else 0.0,
                "Words": f"{budget.used/1e6:.1f}M"
            }
            logger.update_progress(step, **eval_metrics)
            
            # Use the enhanced metric display
            bl_metric = logger.metric("BLiMP", f"{bl:.3f}", good_threshold=0.7, warn_threshold=0.5)
            mo_score = mo0 if 'mo0' in locals() else 0.0
            mo_metric = logger.metric("Morph", f"{mo_score:.3f}", good_threshold=0.7, warn_threshold=0.5)
            
            logger.eval(step, f"{bl_metric}, {mo_metric}")
            
            # Save checkpoint
            save_path = os.path.join(args.save_dir, f"step_{step}")
            student.save_pretrained(save_path)
            logger.success(f"💾 Saved checkpoint: {save_path}")

        if step >= args.steps:
            logger.success("🎉 Reached maximum steps. Training complete!")
            return  # Exit the training loop
    
    # Close progress bar
    logger.close_progress_bar()
    
    # Final evaluation and summary
    logger.info("🏁 Running final evaluation...")
    final_bl = 0.0
    final_mo = mini_morph(student, tokenizer)
    
    # Calculate improvements (with fallback if initial eval was skipped)
    bl_improvement = final_bl - bl0 if 'bl0' in locals() else 0.0
    mo_improvement = final_mo - mo0 if 'mo0' in locals() else 0.0
    
    # Create summary using the enhanced logger
    summary_data = {
        "Final BLiMP": (f"{final_bl:.3f} (Δ{bl_improvement:+.3f})", MLColors.improvement_colors(bl_improvement)),
        "Final Morph": (f"{final_mo:.3f} (Δ{mo_improvement:+.3f})", MLColors.improvement_colors(mo_improvement)),
        "Total Steps": (f"{step:,}", None),
        "Words Processed": (f"{budget.used/1e6:.1f}M / {budget.limit/1e6:.0f}M", None),
        "Training Time": (f"{time.time() - logger.start_time:.1f}s", None),
    }
    
    logger.summary_table(summary_data, "🎉 TRAINING COMPLETE")
    
    # Save final model
    final_path = os.path.join(args.save_dir, "final")
    student.save_pretrained(final_path)
    logger.success(f"💾 Final model saved: {final_path}")


if __name__ == "__main__":
    main()

    # # Optional: compile models (PyTorch 2.x) after first dummy forward for stable shapes
    # try:
    #     student = torch.compile(student, mode="reduce-overhead", fullgraph=False)
    #     reference = torch.compile(reference, mode="reduce-overhead", fullgraph=False)
    #     logger.info("Enabled torch.compile for student & reference.")
    # except Exception as e:
    #     logger.warning(f"torch.compile skipped: {e}")




