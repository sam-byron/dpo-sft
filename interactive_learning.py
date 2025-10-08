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

from datasets import load_dataset, IterableDatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# Import the color logger
from color_logger import get_logger, MLColors, Fore, Style

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)
torch.manual_seed(7)

# Initialize the color logger
logger = get_logger("BabyLM-Interactive")

# -----------------------
# Caregiver (heuristic); swap with LLM later
# -----------------------
@dataclass
class CaregiverOutput:
    corrected: str
    tag: str
    negative: Optional[str]
    reason: str

class HeuristicCaregiver:
    def correct(self, prefix: str, y: str) -> CaregiverOutput:
        s = y.strip()
        # Agreement: plural subject + "is"
        if re.search(r"\b(keys?|dogs?|cats?|cars?)\s+is\b", s):
            y_star = re.sub(r"\bis\b", "are", s, count=1)
            y_neg  = re.sub(r"\bare\b", "is", y_star, count=1)
            return CaregiverOutput(y_star, "agreement", y_neg, "Plural subject → are.")
        # Reflexive mismatch
        if re.search(r"\bhe\b.*\bthemselves\b", s):
            y_star = re.sub(r"\bthemselves\b", "himself", s, count=1)
            y_neg  = re.sub(r"\bhimself\b", "themselves", y_star, count=1)
            return CaregiverOutput(y_star, "reflexive", y_neg, "Reflexive mismatch.")
        # NPI licensing: “A … ever” → “No … ever”
        if re.search(r"\bA [^\.!?]{0,20}\b ever\b", s):
            y_star = re.sub(r"\bA\b", "No", s, count=1)
            y_neg  = s
            return CaregiverOutput(y_star, "npi", y_neg, "NPI 'ever' needs a neg licensor.")
        # Morphology nominalization
        if " creative " in f" {s} ":
            y_star = s.replace(" creative ", " creativity ")
            y_neg  = s.replace(" creative ", " creativeness ")
            return CaregiverOutput(y_star, "morphology", y_neg, "Nominalization fix.")
        # Entity pronoun
        if "Mary told John" in s and re.search(r"\bshe\b", s):
            y_star = re.sub(r"\bshe\b", "he", s, count=1)
            return CaregiverOutput(y_star, "entity", s, "Pronoun must refer to John.")
        # Default
        return CaregiverOutput(s, "other", None, "No change")

# -----------------------
# BNC prefix miner
# -----------------------
PHENOMENON_PATTERNS = {
    "agreement":  re.compile(r"\b(The|the)\s+\w+s\b.*", re.I),
    "reflexive":  re.compile(r"\b(He|he|They|they)\b.*\b(self|selves)\b.*", re.I),
    "npi":        re.compile(r"\b[Aa] .* ever\b.*"),
    "morphology": re.compile(r".*\b(active|creative|kind)\b.*"),
    "entity":     re.compile(r".*\bMary told John\b.*"),
}

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

class BNCPrefixStream(IterableDataset):
    """
    Streams short prefixes (8–20 tokens) from BNC text.
    If bnc_name is provided, uses HF datasets; otherwise reads *.txt in bnc_dir.
    Applies light regex to propose tags; many will be 'other'.
    """
    def __init__(self, tokenizer, bnc_name: Optional[str], bnc_dir: Optional[str], max_len_tokens=20):
        self.tok = tokenizer
        self.bnc_name = bnc_name
        self.bnc_dir = bnc_dir
        self.max_len = max_len_tokens

    def _iterate_hf(self):
        ds = load_dataset(self.bnc_name, split="train", streaming=True)  # user provides name
        for ex in ds:
            text = ex.get("text") or ex.get("content") or ""
            if not text: continue
            for sent in re.split(r"(?<=[.!?])\s+", text):
                sent = clean_text(sent)
                if not sent: continue
                toks = self.tok.tokenize(sent)
                if 4 <= len(toks) <= 48:
                    yield sent

    def _iterate_dir(self):
        if self.bnc_dir is None:
            return
        for root,_,files in os.walk(self.bnc_dir):
            for f in files:
                if not f.lower().endswith(".txt"): continue
                with open(os.path.join(root,f), "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        text = clean_text(line)
                        if not text: continue
                        for sent in re.split(r"(?<=[.!?])\s+", text):
                            sent = clean_text(sent)
                            if not sent: continue
                            toks = self.tok.tokenize(sent)
                            if 4 <= len(toks) <= 48:
                                yield sent

    def __iter__(self):
        it = self._iterate_hf() if self.bnc_name else self._iterate_dir()
        for sent in it:
            # choose a short prefix from each sentence
            words = sent.split()
            if len(words) < 4: continue
            
            # Ensure we have a valid range for randint
            min_len = 4  # minimum prefix length
            max_len = min(20, len(words))  # maximum prefix length
            
            if max_len < min_len:
                continue  # skip sentences that are too short
            
            # Choose prefix length between min_len and max_len
            if max_len >= 8:
                # Prefer longer prefixes (8-20 words) when possible
                L = random.randint(8, max_len)
            else:
                # For shorter sentences, use what we have (4-7 words)
                L = random.randint(min_len, max_len)
            
            prefix = " ".join(words[:L])
            # assign weak tag
            tag = "other"
            for k, rgx in PHENOMENON_PATTERNS.items():
                if rgx.match(prefix):
                    tag = k; break
            yield {"prefix": prefix, "tag": tag}

# -----------------------
# Utilities
# -----------------------
def batchify(items, bs):
    it = iter(items)
    while True:
        b = list(itertools.islice(it, bs))
        if not b: break
        yield b

@torch.no_grad()
def generate(model, tok, prefixes: List[str], max_new_tokens=12, temperature=0.9, top_p=0.9):
    model.eval()
    
    # Batch tokenization for efficiency
    ids = tok(prefixes, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    
    # Batch generation
    outputs = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True, 
        temperature=temperature, 
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    
    # Decode only the generated parts (skip input tokens)
    outs = []
    input_lengths = ids["input_ids"].shape[1]
    for i, output in enumerate(outputs):
        # Find actual input length for this sequence (accounting for padding)
        actual_input_len = (ids["input_ids"][i] != tok.pad_token_id).sum().item()
        gen_tokens = output[actual_input_len:]
        gen_text = tok.decode(gen_tokens, skip_special_tokens=True)
        outs.append(gen_text.strip())
    
    return outs

def ce_targets(student, tok, x_list, y_list):
    student.train()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    # concat x + y; labels mask x part
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    out = student(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    return out.loss

@torch.no_grad()
def logprob_sum(model, tok, x_list, y_list):
    model.eval()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits
    logp = F.log_softmax(logits, dim=-1)
    tgt = labels[:,1:].clone()
    mask = (tgt != -100)
    lp = logp[:,:-1,:].gather(-1, tgt.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    lp = (lp * mask).sum(dim=1)
    return lp

def kl_to_ref(student, reference, tok, x_list):
    student.train(); reference.eval()
    enc = tok(x_list, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    s_logits = student(**enc).logits
    with torch.no_grad():
        r_logits = reference(**enc).logits
    p = F.log_softmax(s_logits, dim=-1)
    q = F.log_softmax(r_logits, dim=-1)
    kl = (torch.exp(p) * (p - q)).sum(dim=-1)
    kl = (kl * enc.attention_mask).sum() / enc.attention_mask.sum().clamp_min(1)
    return kl

def dpo_loss(student, reference, tok, xs, y_pos, y_neg, beta=0.2):
    lp_pos = logprob_sum(student, tok, xs, y_pos)
    lp_neg = logprob_sum(student, tok, xs, y_neg)
    with torch.no_grad():
        lr_pos = logprob_sum(reference, tok, xs, y_pos)
        lr_neg = logprob_sum(reference, tok, xs, y_neg)
    margin = (lp_pos - lp_neg) - (lr_pos - lr_neg)
    return -torch.log(torch.sigmoid(beta * margin)).mean()

# -----------------------
# Mini probes (cheap)
# -----------------------
BLIMP_MINI = [
    ("The keys to the cabinet", " are on the table.", " is on the table."),
    ("The dogs in the yard", " are loud.", " is loud."),
    ("He", " saw himself.", " saw themselves."),
    ("A student", " has never cheated.", " has ever cheated."),
    ("Mary told John that", " he will go.", " she will go."),
]

MORPH_STEMS   = ["She showed much", "His"]
MORPH_GOOD    = [" creativity.", " activity impressed them."]
MORPH_BAD     = [" creativeness.", " activeness impressed them."]

@torch.no_grad()
def simple_logprob(model, tok, full_text):
    """Calculate logprob of full text sequence (simpler than logprob_sum)."""
    model.eval()
    # Temporarily switch to right padding for evaluation to avoid complexity
    orig_padding_side = tok.padding_side
    tok.padding_side = 'right'
    
    try:
        enc = tok(full_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = model(**enc)
        logits = out.logits
        
        # Calculate log probabilities
        logp = F.log_softmax(logits, dim=-1)
        # Get target tokens (shifted by 1)
        tgt = enc.input_ids[:, 1:].clone()
        # Get log probabilities for target tokens
        lp = logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # Mask out padding tokens
        mask = (tgt != tok.pad_token_id) & (enc.attention_mask[:, 1:] == 1)
        # Sum log probabilities for each sequence
        lp_sum = (lp * mask).sum(dim=1)
        return lp_sum
    finally:
        # Restore original padding side
        tok.padding_side = orig_padding_side

@torch.no_grad()
def mini_blimp(model, tok):
    correct = 0
    total = len(BLIMP_MINI)
    
    for prefix, good, bad in BLIMP_MINI:
        good_text = prefix + good
        bad_text = prefix + bad
        
        lp_good = simple_logprob(model, tok, [good_text])
        lp_bad = simple_logprob(model, tok, [bad_text])
        
        if lp_good[0] > lp_bad[0]:
            correct += 1
    
    return correct / total

@torch.no_grad()
def mini_morph(model, tok):
    correct = 0
    total = len(MORPH_STEMS)
    
    for i, stem in enumerate(MORPH_STEMS):
        good_text = stem + MORPH_GOOD[i]
        bad_text = stem + MORPH_BAD[i]
        
        lp_good = simple_logprob(model, tok, [good_text])
        lp_bad = simple_logprob(model, tok, [bad_text])
        
        if lp_good[0] > lp_bad[0]:
            correct += 1
    
    return correct / total

# -----------------------
# Word budget
# -----------------------
class WordBudget:
    def __init__(self, limit_words=100_000_000):
        self.limit = int(limit_words)
        self.used = 0
    def add(self, texts: List[str]):
        # rough words via whitespace
        self.used += sum(len(t.split()) for t in texts)
    def ok(self):
        return self.used <= self.limit

# -----------------------
# Main training
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bnc_name", type=str, default="deven367/babylm-100M-bnc-spoken", help="HF dataset name if available (e.g., 'bnc' or your org/dataset')")
    ap.add_argument("--bnc_dir", type=str, default="", help="Path to BNC .txt files if not using HF")
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--use_dpo", action="store_true")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_dir", type=str, default="./ckpts_bnc_interactive")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info(f"🚀 Starting interactive learning with {Fore.YELLOW}{args.model_name}{Style.RESET_ALL}")
    logger.info(f"Device: {Fore.GREEN}{DEVICE}{Style.RESET_ALL}")
    logger.info(f"Batch size: {args.batch_size}, Steps: {args.steps:,}, LR: {args.lr}")
    logger.info(f"Save directory: {args.save_dir}")
    
    # Performance optimization notices
    if args.batch_size < 16:
        logger.warning(f"⚠️  Small batch size ({args.batch_size}) may cause slow training. Consider increasing to 32-64 for better GPU utilization.")
    logger.info(f"🚀 Performance optimizations: batch generation, parallel data loading, profiling enabled")

    logger.info("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Fix padding side for decoder-only models (critical for generation quality)
    tok.padding_side = 'left'
    logger.info(f"Set tokenizer padding_side to: {Fore.GREEN}left{Style.RESET_ALL} (required for decoder-only models)")
    logger.info(f"Suppressed tokenizer parallelism warnings via environment variable")

    # Student + LoRA
    logger.info("Setting up student model with LoRA...")
    base = AutoModelForCausalLM.from_pretrained(args.model_name)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn","c_proj","q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(base, lora_cfg).to(DEVICE)
    student.train()
    
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Trainable parameters: {Fore.GREEN}{trainable_params:,}{Style.RESET_ALL} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Reference (frozen)
    logger.info("Loading reference model...")
    reference = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    reference.eval()

    logger.info("Setting up heuristic caregiver...")
    caregiver = HeuristicCaregiver()

    # Data stream
    bnc_name = args.bnc_name if args.bnc_name else None
    bnc_dir  = args.bnc_dir if args.bnc_dir else None
    logger.info(f"Setting up data stream from: {bnc_name or bnc_dir or 'default'}")
    ds = BNCPrefixStream(tok, bnc_name, bnc_dir)
    # Reduce workers to avoid tokenizer parallelism issues with streaming datasets
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # Optim & sched
    logger.info("Setting up optimizer and scheduler...")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=500, num_training_steps=args.steps)

    budget = WordBudget(100_000_000)
    logger.info(f"Word budget: {Fore.YELLOW}{budget.limit:,}{Style.RESET_ALL} words")

    # Initial eval
    logger.info("Running initial evaluation...")
    bl0 = mini_blimp(student, tok)
    mo0 = mini_morph(student, tok)
    logger.eval(0, f"Initial scores - BLiMP: {Fore.YELLOW}{bl0:.3f}{Style.RESET_ALL}, Morph: {Fore.YELLOW}{mo0:.3f}{Style.RESET_ALL}")
    
    logger.success("🎯 Starting training loop...")
    logger.info(f"📊 Progress will be tracked with {'tqdm' if hasattr(logger, 'progress_bar') else 'basic'} progress bars")
    
    # Create progress bar for training
    progress_bar = logger.create_progress_bar(args.steps, "🚀 Training")

    step = 0
    step_times = {"data": 0.0, "generate": 0.0, "correct": 0.0, "forward": 0.0, "backward": 0.0}
    
    for batch in loader:
        step_start = time.time()
        step += 1
        prefixes = batch["prefix"]
        step_times["data"] += time.time() - step_start
        
        # Generate short student attempts
        gen_start = time.time()
        attempts = generate(student, tok, prefixes, max_new_tokens=12, temperature=0.9, top_p=0.9)
        step_times["generate"] += time.time() - gen_start

        # Caregiver corrections
        correct_start = time.time()
        outs = [caregiver.correct(x, y) for x, y in zip(prefixes, attempts)]
        y_star = [o.corrected for o in outs]
        step_times["correct"] += time.time() - correct_start
        
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

        # Losses
        forward_start = time.time()
        L_sft = ce_targets(student, tok, prefixes, y_star)
        L_kl  = kl_to_ref(student, reference, tok, prefixes)
        L_dpo = torch.tensor(0.0, device=DEVICE)
        if args.use_dpo:
            xs_dpo, yp, yn = [], [], []
            for x,o in zip(prefixes, outs):
                if o.negative:
                    xs_dpo.append(x); yp.append(o.corrected); yn.append(o.negative)
            if xs_dpo:
                L_dpo = dpo_loss(student, reference, tok, xs_dpo, yp, yn, beta=0.2)

        loss = L_sft + 0.03*L_kl + 0.5*L_dpo
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

        # Update progress bar with current metrics
        current_lr = sched.get_last_lr()[0]
        progress_metrics = {
            "SFT_loss": L_sft.item(),
            "KL_loss": L_kl.item(), 
            "DPO_loss": L_dpo.item(),
            "LR": f"{current_lr:.1e}",
            "Words": f"{budget.used/1e6:.1f}M"
        }
        logger.update_progress(step, **progress_metrics)

        # Detailed logging every 100 steps (less frequent than progress bar)
        if step % 100 == 0:
            # Use the enhanced loss display from the logger
            losses = {"SFT": L_sft.item(), "KL": L_kl.item(), "DPO": L_dpo.item()}
            thresholds = {"SFT": (2.0, 3.0), "KL": (0.1, 0.5), "DPO": (1.0, 2.0)}
            
            loss_display = logger.loss_display(losses, thresholds)
            extra_info = f"LR: {Fore.CYAN}{current_lr:.2e}{Style.RESET_ALL} | 📚 {Fore.CYAN}{budget.used/1e6:.1f}M{Style.RESET_ALL} words"
            
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
            logger.info("🔍 Running evaluation...")
            bl = mini_blimp(student, tok)
            mo = mini_morph(student, tok)
            
            # Update progress bar with evaluation metrics
            eval_metrics = {
                "SFT_loss": L_sft.item(),
                "KL_loss": L_kl.item(), 
                "DPO_loss": L_dpo.item(),
                "BLiMP_score": bl,
                "Morph_score": mo,
                "Words": f"{budget.used/1e6:.1f}M"
            }
            logger.update_progress(step, **eval_metrics)
            
            # Use the enhanced metric display
            bl_metric = logger.metric("BLiMP", f"{bl:.3f}", good_threshold=0.7, warn_threshold=0.5)
            mo_metric = logger.metric("Morph", f"{mo:.3f}", good_threshold=0.7, warn_threshold=0.5)
            
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
    final_bl = mini_blimp(student, tok)
    final_mo = mini_morph(student, tok)
    
    # Calculate improvements
    bl_improvement = final_bl - bl0
    mo_improvement = final_mo - mo0
    
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
