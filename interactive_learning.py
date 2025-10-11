# bnc_interactive_train.py
import os, re, math, random, json, argparse, itertools, time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, PeftModel
from load_lora import load_and_merge_lora, load_adapter_config

# Import the color logger
from color_logger import get_logger, MLColors, Fore, Style
# BLiMP evaluation utilities
from blimp import run_subset, ensure_subsets_list, pick_split

from interactive_utils import Caregiver, CaregiverOutput, BNCPrefixStream, dpo_loss, eval_blimp_hf, get_uncertainty, generate, ce_targets, kl_to_ref, logprob_sum, mini_morph, WordBudget, combined_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)
torch.manual_seed(7)

# Add after imports & seeds (near top, after DEVICE):
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Initialize the color logger
logger = get_logger("BabyLM-Interactive")

# # Make compile robust to dynamic shapes and fall back instead of crashing
# try:
#     import torch._dynamo as dynamo
#     import torch._inductor.config as inductor_config
    
#     # Configure dynamo for better dynamic shape handling
#     dynamo.config.dynamic_shapes = True
#     dynamo.config.suppress_errors = True
#     dynamo.config.verbose = False  # Reduce verbosity
    
#     # Configure CUDAGraph settings - be more permissive
#     inductor_config.triton.cudagraph_skip_dynamic_graphs = True
#     inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None  # Silence warnings
    
#     # Additional configs to improve compilation success
#     inductor_config.max_autotune = False  # Faster compile, good enough perf
#     inductor_config.triton.cudagraphs = False  # Disable cudagraphs entirely (they're causing the warnings)
    
#     logger.info("Configured torch.compile settings for dynamic shapes")
# except Exception as e:
#     logger.warning(f"torch._dynamo/inductor config not available: {e}")

from datasets import load_dataset, get_dataset_config_names

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
    args = ap.parse_args()

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
        base_model_name = args.model_name
        try:
            cfg = load_adapter_config(os.path.join(args.model_path, "adapter_config.json"))
            base_model_name = cfg.get("base_model_name_or_path", base_model_name)
        except Exception as e:
            logger.warning(f"Could not read adapter_config.json: {e}; falling back to --model_name={base_model_name}")

        # Tokenizer
        tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'left'

        # Base + attach adapters without merging
        base = AutoModelForCausalLM.from_pretrained(base_model_name)
        student = PeftModel.from_pretrained(base, args.model_path)
        student.to(DEVICE)

        # Freeze base weights; train only LoRA adapter params
        for n, p in student.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False
        student.train()
        logger.info("Loaded base + LoRA; adapters set trainable, base frozen")
    else:
        logger.info("Loading tokenizer...")
        # Use fast tokenizer explicitly for better performance
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        # Fix padding side for decoder-only models (critical for generation quality)
        tok.padding_side = 'left'  # This should be the ONLY place we set padding_side
        logger.info(f"Set tokenizer padding_side to: {Fore.GREEN}left{Style.RESET_ALL} (required for decoder-only models)")
        logger.info(f"Using fast tokenizer: {Fore.GREEN}{tok.is_fast}{Style.RESET_ALL}")
        logger.info(f"Suppressed tokenizer parallelism warnings via environment variable")

        # Student + LoRA (fresh init)
        logger.info("Setting up student model with LoRA...")
        base = AutoModelForCausalLM.from_pretrained(args.model_name)
        lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn","c_proj","q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        student = get_peft_model(base, lora_cfg).to(DEVICE)
        student.train()
    
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Trainable parameters: {Fore.GREEN}{trainable_params:,}{Style.RESET_ALL} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    import copy
    # Reference (frozen)
    logger.info("Loading reference model...")
    reference = copy.deepcopy(student)
    reference.requires_grad_(False)
    reference.to(DEVICE)
    reference.eval()

    # Do NOT compile student (training graph + LoRA + dynamic shapes → unstable)
    # Only compile eval-time models with safe fallbacks
    # try:
    #     reference = torch.compile(reference, mode="reduce-overhead", fullgraph=False, dynamic=True)
    #     logger.info("Enabled torch.compile for reference.")
    # except Exception as e:
    #     logger.warning(f"torch.compile (reference) skipped: {e}")

    logger.info("Setting up caregiver...")
    caregiver = Caregiver()
    # try:
    #     caregiver.model = torch.compile(caregiver.model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    #     logger.info("Enabled torch.compile for caregiver model.")
    # except Exception as e:
    #     logger.warning(f"torch.compile (caregiver) skipped: {e}")

    # Data stream
    bnc_name = args.bnc_name if args.bnc_name else None
    bnc_dir  = args.bnc_dir if args.bnc_dir else None
    logger.info(f"Setting up data stream from: {bnc_name or bnc_dir or 'default'}")
    ds = BNCPrefixStream(tok, bnc_name, bnc_dir)
    # Reduce workers to avoid tokenizer parallelism issues with streaming datasets
    # Add drop_last=True to stabilize batch shape for compiled graphs
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)

    # Optim & sched
    logger.info("Setting up optimizer and scheduler...")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=500, num_training_steps=args.steps)

    budget = WordBudget(100_000_000)
    logger.info(f"Word budget: {Fore.YELLOW}{budget.limit:,}{Style.RESET_ALL} words")

    logger.info("Running initial evaluation on REAL BLiMP…")
    # bl0, per_cat0 = eval_blimp_hf(student, tok, n_per_cat=200, max_len=64, progress=True)
    # mo0 = mini_morph(student, tok)
    # pretty_print_blimp(per_cat0)
    # logger.eval(0, f"Initial BLiMP (real): {bl0:.3f}")

    
    logger.success("🎯 Starting training loop...")
    logger.info(f"📊 Progress will be tracked with {'tqdm' if hasattr(logger, 'progress_bar') else 'basic'} progress bars")
    
    # Create progress bar for training
    progress_bar = logger.create_progress_bar(args.steps, "🚀 Training")

    step = 0
    step_times = {"data": 0.0, "generate": 0.0, "correct": 0.0, "forward": 0.0, "backward": 0.0}
    
    for batch in loader:
        step_start = time.time()
        step += 1
        
        # Up-sample phenomenon-rich prefixes for 50% of each batch
        if "tag" in batch:
            rich_idx = [i for i,t in enumerate(batch["tag"]) if t in ("agreement","reflexive","npi","morphology","entity")]
            if len(rich_idx) >= args.batch_size // 2:
                sel = rich_idx[:args.batch_size//2] + list(range(args.batch_size//2))
                prefixes = [batch["prefix"][i] for i in sel]
            else:
                prefixes = batch["prefix"]
        else:
            prefixes = batch["prefix"]

        # Light nudge: append a small cue token to expose the locus (helps heuristics)
        prefixes = [p + " is" if re.search(r"\b\w+s\b$", p) else p for p in prefixes]

        # Generate short student attempts
        gen_start = time.time()
        # Adaptive decode budget (shorter contexts → fewer new tokens)
        avg_len = sum(len(p.split()) for p in prefixes)/len(prefixes)
        adaptive_new = 8 if avg_len < 10 else 12
        attempts = generate(student, tok, prefixes, max_new_tokens=adaptive_new, temperature=0.9, top_p=0.9)
        # Ensure strings for downstream typing (guard lints)
        attempts = [a if isinstance(a, str) else "" for a in attempts]
        gen_time = time.time() - gen_start
        step_times["generate"] += gen_time
        
        # NEW: Calculate uncertainty and select samples for correction
        correct_start = time.time()
        uncertainties = get_uncertainty(student, tok, prefixes, attempts)

        # Correct top 30% most uncertain samples
        threshold = sorted(uncertainties, reverse=True)[int(0.3 * len(uncertainties))]
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
                
                # Merge back: corrected samples + unchanged samples
                outs = []
                correct_iter = iter(outs_subset)
                for i, flag in enumerate(needs_correction):
                    if flag:
                        outs.append(next(correct_iter))
                    else:
                        # Keep student's original output
                        outs.append(CaregiverOutput(attempts[i], "other", None, "no_correction"))
            except Exception as e:
                logger.warning(f"⚠️  Caregiver batch failed: {e}")
                outs = [CaregiverOutput(str(att or ""), "other", None, "caregiver_error") for att in attempts]
        else:
            # No samples need correction
            outs = [CaregiverOutput(str(att or ""), "other", None, "no_correction") for att in attempts]
        
        y_star = [o.corrected for o in outs]
        correct_time = time.time() - correct_start
        step_times["correct"] += correct_time

        # Track correction statistics for logging
        if step % 200 == 0:  # Less frequent than main logging
            tag_counts = {}
            for o in outs:
                tag_counts[o.tag] = tag_counts.get(o.tag, 0) + 1
            
            if any(tag != "other" for tag in tag_counts):
                corrections_msg = []
                for tag, count in sorted(tag_counts.items()):
                    if tag != "other" and count > 0:
                        color = Fore.GREEN if tag in ["agreement", "reflexive"] else Fore.CYAN
                        corrections_msg.append(f"{color}{tag}: {count}{Style.RESET_ALL}")
                
                if corrections_msg:
                    logger.debug(f"📝 Corrections in batch: {', '.join(corrections_msg)}")

        # --- Compute core losses first ---
        forward_start = time.time()
        L_sft = ce_targets(student, tok, prefixes, y_star)
        L_kl  = kl_to_ref(student, reference, tok, prefixes, attempts)  # ✅ Use attempts, not y_star
        L_dpo = torch.tensor(0.0, device=DEVICE)  # default (in case we skip or no pairs)

        # --- DPO block (optional) ---
        if args.use_dpo:
            xs_dpo, yp, yn = [], [], []
            for x, o in zip(prefixes, outs):
                if not o.negative:
                    continue
                # filter by reference preference (only strong contrastive pairs)
                lp_pos_ref = logprob_sum(reference, tok, [x], [o.corrected])[0]
                lp_neg_ref = logprob_sum(reference, tok, [x], [o.negative])[0]
                # if (lp_pos_ref - lp_neg_ref).item() < 0.5:
                #     continue
                xs_dpo.append(x)
                yp.append(o.corrected)
                yn.append(o.negative)

            if xs_dpo:
                L_dpo = dpo_loss(student, reference, tok, xs_dpo, yp, yn, beta=2.0)

        # --- Combine losses (final total) ---
        dpo_weight = 1.0 if args.use_dpo else 0.0
        loss = L_sft + 0.03 * L_kl + dpo_weight * L_dpo
        step_times["forward"] += time.time() - forward_start

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
        
        # Use dictionary method for progress bar updates
        progress_metrics = {
            "Loss": f"{loss.item():.3f}",
            "SFT": f"{L_sft.item():.3f}",
            "KL": f"{L_kl.item():.3f}", 
            "DPO": f"{L_dpo.item():.3f}",
            "LR": f"{current_lr:.1e}",
            "Corr": f"{n_correct}/{len(prefixes)}",
            "Words": f"{budget.used/1e6:.1f}M"
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
            bl, per_cat = eval_blimp_hf(student, tok, n_per_cat=200, max_len=64, progress=False)
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
            break
    
    # Close progress bar
    logger.close_progress_bar()
    
    # Final evaluation and summary
    logger.info("🏁 Running final evaluation...")
    final_bl = 0.0
    final_mo = mini_morph(student, tok)
    
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

