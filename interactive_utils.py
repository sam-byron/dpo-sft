# interactive_utils.py
# ==================================================================================
# PERFORMANCE OPTIMIZATIONS:
# ==================================================================================
# - All inference functions use @torch.no_grad() or torch.inference_mode()
# - Mixed precision (bfloat16) via torch.autocast for all forward passes
# - KV-cache enabled for generation
# - Bucket-based padding to reduce recompilations
# - Efficient log-probability calculations with proper masking
# - Left padding for batch generation efficiency
# ==================================================================================
import os
import re
import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, cast
import itertools

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, AutoConfig
from peft import LoraConfig, get_peft_model

# Import the color logger
from color_logger import get_logger, MLColors, Fore, Style
# BLiMP evaluation utilities
from blimp import run_subset, ensure_subsets_list, pick_split

MAX_CONCAT = 512  # enough for short prefixes + short corrections
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# @torch.no_grad()
def get_uncertainty(student, tok, prefixes, attempts):
    """Return per-sample uncertainty (entropy) as selection criterion (inference only)."""
    student.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        enc_x = tok(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
        enc_y = tok(attempts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
        
        # Move to device and ensure Long dtype in one step
        input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long().to(DEVICE)
        attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long().to(DEVICE)
        
        out = student(input_ids=input_ids[:, :MAX_CONCAT], attention_mask=attn_mask[:, :MAX_CONCAT])
    
    logits = out.logits[:, enc_x.input_ids.shape[1]-1:-1, :]
    probs = F.softmax(logits, dim=-1)
    # Clamp to avoid log(0)
    entropy = -(probs * torch.clamp(probs.log(), min=-100)).sum(dim=-1).mean(dim=-1)
    student.train()
    return entropy.detach().cpu().tolist()

def force_json(text: str) -> Dict:
    # Extract first {...} block to guard against stray tokens
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in caregiver output")
    js = json.loads(m.group(0))
    return js

def dpo_loss(student, reference, tok, xs, y_pos, y_neg, beta=0.1, max_len=256):
    """
    Direct Preference Optimization (Rafailov et al. 2023):
      L = - E[ log σ( β ( log πθ(y+|x) - log πθ(y-|x) - (log πref(y+|x) - log πref(y-|x)) ) ) ]
    We approximate log π(y|x) with sum of token log-probs over y conditioned on x.
    Args:
      xs, y_pos, y_neg: lists of strings (same length)
      beta: temperature (larger -> more aggressive)
    Returns: scalar loss
    """
    student.train()
    reference.eval()

    # Student preferences (requires grad)
    lp_pos_stu = logprob_sum_with_grad(student, tok, xs, y_pos, max_len=max_len)  # [B]
    lp_neg_stu = logprob_sum_with_grad(student, tok, xs, y_neg, max_len=max_len)  # [B]
    stu_delta = lp_pos_stu - lp_neg_stu
    
    # Reference preferences (detached, computed once)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        lp_pos_ref = logprob_sum(reference, tok, xs, y_pos, max_len=max_len)  # [B]
        lp_neg_ref = logprob_sum(reference, tok, xs, y_neg, max_len=max_len)  # [B]
        ref_delta = (lp_pos_ref - lp_neg_ref).detach()  # Explicit detach for safety

    # Score and logistic loss
    margin = beta * (stu_delta - ref_delta)
    # log σ(z) = -softplus(-z); we want -E[log σ(margin)]
    loss = F.softplus(-margin).mean()
    return loss


@dataclass
class CaregiverOutput:
    corrected: str


ALLOWED_TAGS = {"agreement","reflexive","npi","entity","morphology","other"}

def parse_caregiver_text(text: str, student_fallback: str) -> CaregiverOutput:
    """
    Parse caregiver output from simple text format.
    Expected format (pipe-delimited):
    CORRECTED: <text>
    TAG: <tag>
    NEGATIVE: <text or NONE>
    REASON: <text>
    
    Falls back gracefully if format is malformed.
    """
    
    return CaregiverOutput(" ".join(text.split()))

def is_causal_lm(model_name: str) -> bool:
    """Check if a model is a causal language model based on its config."""
    try:
        config = AutoConfig.from_pretrained(model_name)
        # Check if it's a decoder-only (causal) model
        return getattr(config, 'is_decoder', True) and not getattr(config, 'is_encoder_decoder', False)
    except Exception:
        return False

def is_seq2seq(model_name: str) -> bool:
    """Check if a model is a sequence-to-sequence (encoder-decoder) model."""
    try:
        config = AutoConfig.from_pretrained(model_name)
        # Check if it's an encoder-decoder model
        return getattr(config, 'is_encoder_decoder', False)
    except Exception:
        return False
class Caregiver:
    """Caregiver model that provides text-based corrections without JSON."""
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct', rng_seed: int = 0):

        if is_causal_lm(model_name):
            self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = 'left'

            # Load in bfloat16 directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        elif is_seq2seq(model_name):
            self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = 'right'
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            raise ValueError(f"Model {model_name} is neither a causal LM nor a seq2seq model.") 

        self.model.eval()
        
        # Freeze all parameters for inference-only usage
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_prompts(self, prefixes: List[str], suffixes: List[str]) -> list[str]:
        """Build a simple instruction prompt for CoEdit."""
        # CoEdit expects: "Fix grammar: <text>" or "Make this coherent: <text>"
        prompts = [
            f"Fix grammatical errors: {prefix} {suffix}" for prefix, suffix in zip(prefixes, suffixes)  
        ]
        return prompts

    # Optimize correct_batch to reuse pre-tokenized system and use autocast/inference_mode
    def correct_batch(self, prefixes: List[str], students: List[str]) -> List[CaregiverOutput]:
        # prompt = f"first half:{p} second half:{s}. If this sentence is incorrect, provide a corrected version of the second half such that the sentence requires the minimal number of edits. Provide one sentence only.\nCorrected version:"
        self.model.eval()
        if is_causal_lm(self.model.config._name_or_path):
            prompts = [
            f"first half:{p} second half:{s}. If this sentence is incorrect, provide a corrected " 
            "version of the second half such that the sentence requires the minimal number of edits. You must preserves the first half exactly. "
            "Provide one sentence only.\nCorrected version:"
                for p, s in zip(prefixes, students)
            ]
            chat_payloads = self.tok.apply_chat_template(
                [[
                    {"role": "system", "content": "You are a teacher."},
                    {"role": "user", "content": p}
                ] for p in prompts],
                tokenize=False,
                add_generation_prompt=True,
            )
            tok_chat_payload = self.tok(chat_payloads, return_tensors="pt", padding=True, truncation=True)
            input_ids = tok_chat_payload.input_ids.to(DEVICE)
            attention_mask = tok_chat_payload.attention_mask.to(DEVICE)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                    do_sample=True,
                    # temperature=0.7,
                    # top_p=0.9,
                    # repetition_penalty=1.1,
                    early_stopping=True,
                )
            generated_ids = [
                output_ids[len(input_ids_):] for input_ids_, output_ids in zip(input_ids
            , generated_ids)
            ]
        elif is_seq2seq(self.model.config._name_or_path):
            prompts = self._build_prompts(prefixes, students)
            tok_payload = self.tok(prompts, return_tensors="pt", padding=True)
            # prefixes_ids = self.tok(prefixes, return_tensors="pt", padding=False).input_ids
            input_ids = tok_payload.input_ids.to(DEVICE)
            # attention_mask = torch.cat([p.attention_mask for p in tok_payload]).to(DEVICE)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                    do_sample=True,
                    # temperature=0.7,
                    # top_p=0.9,
                    # repetition_penalty=1.1,
                    # early_stopping=True,
                )
            generated_ids = [
                output_ids[len(self.tok(prefixes_)):] for prefixes_, output_ids in zip(prefixes, generated_ids)
            ]
        else:
            raise ValueError("Model type not supported for correction.")

        # Bucketed padding
        # target_len = pad_to_bucket(chat_payloads, self.tok, buckets=[64, 128, 256])
        # dyn = {k: v.to('cuda') for k, v in tok_chat_payload.items()}
        # tok_chat_payload.to('cuda')
        # Move to device once
        
       

        responses = self.tok.batch_decode(generated_ids, skip_special_tokens=True)
        results = []
        # Parse each response
        for response, student in zip(responses, students):
            results.append(parse_caregiver_text(response, student.strip()))
        
        return results

    def correct(self, prefix: str, student: str) -> CaregiverOutput:
        """Single correction."""
        results = self.correct_batch([prefix], [student])
        return results[0]

# @torch.no_grad()
def eval_blimp_hf(model, tok, n_per_cat=50, max_len=64, categories=None, progress=True):
    """Evaluate on BLiMP using shared helpers from blimp.py.

    Returns (overall_accuracy, per_category_acc_dict).
    """
    device = torch.device(DEVICE)
    if categories is None:
        categories = ensure_subsets_list("blimp")

    per_cat: Dict[str, float] = {}
    total_n = 0
    total_right = 0.0

    for cat in categories:
        try:
            split = pick_split("blimp", cat, "auto")
        except Exception as e:
            if progress:
                print(f"[BLiMP/{cat}] skipped: {e}")
            continue
        res = run_subset(model, tok, cat, split, device, n_per_cat, normalize="none", dump=0)
        per_cat[cat] = float(res.get("acc", 0.0))
        n = int(res.get("n", 0))
        total_n += n
        total_right += per_cat[cat] * n
        if progress:
            print(f"[BLiMP/{cat:>28}] acc={per_cat[cat]:.3f}  n={n}")

    overall = (total_right / max(1, total_n)) if total_n > 0 else 0.0
    return overall, per_cat

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def batchify(items, bs):
    it = iter(items)
    while True:
        b = list(itertools.islice(it, bs))
        if not b: break
        yield b

# @torch.no_grad()
def complete(model, tok, prefixes, max_new_tokens=20, temperature=1.0, top_p=None, fast_sample=False):
    """Generate continuations for prefixes using the model (inference only)."""
    model.eval()
    
    # Use standard padding (left-side) without bucket optimization for now
    # The bucket optimization was causing attention mask issues
    enc = tok(
        prefixes, 
        return_tensors="pt", 
        padding=True,  # Use dynamic padding instead of max_length
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # Debugging: verify padding is correct (1% sample rate)
    # if torch.rand(1).item() < 0.01:
    #     print(f"[DEBUG] First input_ids: {input_ids[0][:20].tolist()}")
    #     print(f"[DEBUG] First attn_mask: {attention_mask[0][:20].tolist()}")
    #     # Decode only non-padding tokens
    #     non_pad_ids = input_ids[0][attention_mask[0].bool()]
    #     print(f"[DEBUG] Decoded prefix: {tok.decode(non_pad_ids, skip_special_tokens=True)[:50]}")
    
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        if not fast_sample:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        else:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                 # FASTEST SETTINGS:
                do_sample=False,              # Greedy decoding (fastest)
                num_beams=1,                  # No beam search
                use_cache=True,               # KV-cache (essential!)
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

    # Extract only the generated tokens (not the input)
    out = [
        output_ids[len(input_ids_):] for input_ids_, output_ids in zip(input_ids, out)
    ]
    return [tok.decode(g, skip_special_tokens=True).strip() for g in out]

def ce_targets(student, tok, x_list, y_list):
    """Cross-entropy loss on target y given prefix x (requires gradients)."""
    student.train()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    
    # Efficient: concatenate then move to device + cast to long in one operation
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long().to(DEVICE)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long().to(DEVICE)
    
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    
    out = student(
        input_ids=input_ids[:, :MAX_CONCAT],
        attention_mask=attn_mask[:, :MAX_CONCAT],
        labels=labels[:, :MAX_CONCAT]
    )
    return out.loss

# logprob_sum: remove incorrect divide by len(tok) and use autocast
# @torch.no_grad()
def logprob_sum(model, tok, x_list, y_list, max_len=256):
    """Compute sum of log probabilities for y given x (inference only)."""
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
        enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
        
        # Move to device and ensure Long dtype in one step
        input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long().to(DEVICE)
        attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long().to(DEVICE)
        
        labels = input_ids.clone()
        labels[:, :enc_x.input_ids.shape[1]] = -100
        
        out = model(input_ids=input_ids[:, :max_len], attention_mask=attn_mask[:, :max_len])
        logits = out.logits[:, :-1]
        tgt = labels[:, 1:]
        mask_y = (tgt != -100)
        logp_all = F.log_softmax(logits, dim=-1)
        tgt_safe = tgt.masked_fill(~mask_y, 0)
        token_lp = logp_all.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask_y).sum(dim=1)
    
def logprob_sum_with_grad(model, tok, x_list, y_list, max_len=256):
    """Compute sum of log probabilities for y given x (inference only)."""

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
        enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
        
        # Move to device and ensure Long dtype in one step
        input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long().to(DEVICE)
        attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long().to(DEVICE)
        
        labels = input_ids.clone()
        labels[:, :enc_x.input_ids.shape[1]] = -100
        
        out = model(input_ids=input_ids[:, :max_len], attention_mask=attn_mask[:, :max_len])
        logits = out.logits[:, :-1]
        tgt = labels[:, 1:]
        mask_y = (tgt != -100)
        logp_all = F.log_softmax(logits, dim=-1)
        tgt_safe = tgt.masked_fill(~mask_y, 0)
        token_lp = logp_all.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask_y).sum(dim=1)

def kl_to_ref(student, reference, tok, xs, y_pos, y_neg: Optional[List[str]] = None, temperature: float = 1.0):
    """
    Distillation-style KL(student || reference) on target tokens y conditioned on x.
    Student requires gradients, reference doesn't.
    """
    student.train()
    reference.eval()
    
    # Encode x and y (only positives by default)
    enc_x = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    enc_y = tok(y_pos, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    
    # Efficient: concatenate then move to device + cast to long in one operation
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long().to(DEVICE)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long().to(DEVICE)
    
    # Labels to identify y positions (mask x with -100)
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    
    # Student forward WITH gradients (no torch.no_grad!)
    st_out = student(input_ids=input_ids[:, :MAX_CONCAT], attention_mask=attn_mask[:, :MAX_CONCAT])
    
    # Reference forward WITHOUT gradients
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        rf_out = reference(input_ids=input_ids[:, :MAX_CONCAT], attention_mask=attn_mask[:, :MAX_CONCAT])
    
    # Shift for next-token prediction
    st_logits = st_out.logits[:, :-1, :] / temperature
    rf_logits = rf_out.logits[:, :-1, :] / temperature
    mask_y = (labels[:, 1:] != -100)  # positions belonging to y

    # KL(student || reference) over vocab at each token
    log_p = F.log_softmax(st_logits, dim=-1)
    q = F.softmax(rf_logits.detach(), dim=-1)  # Explicit detach for safety

    # Per-token KL: sum over vocab, then mask to y positions
    kl_tok = F.kl_div(log_p, q, reduction="none").sum(dim=-1)  # [B, T]
    kl_tok = kl_tok * mask_y.float()
    denom = mask_y.float().sum().clamp_min(1.0)
    return kl_tok.sum() / denom

# @torch.no_grad()
def simple_logprob(model, tok, full_text):
    """Calculate logprob of full text sequence (simpler than logprob_sum)."""
    model.eval()
    # Temporarily switch to right padding for evaluation
    orig_padding_side = getattr(tok, "padding_side", "right")
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
def mini_morph(model, tok, verbose=False):
    # Minimal morphology probe pairs
    global MORPH_STEMS, MORPH_GOOD, MORPH_BAD
    MORPH_STEMS = [
        "The key",
        "The keys",
        "She",
        "They",
    ]
    MORPH_GOOD = [
        " is on the table.",
        " are on the table.",
        " is happy.",
        " are happy.",
    ]
    MORPH_BAD = [
        " are on the table.",
        " is on the table.",
        " are happy.",
        " is happy.",
    ]
    """
    Same idea: score continuation y given the stem x.
    """
    model.eval()
    correct = 0
    totals = len(MORPH_STEMS)
    details = []

    for stem, good, bad in zip(MORPH_STEMS, MORPH_GOOD, MORPH_BAD):
        lp_g = logprob_sum(model, tok, [stem], [good])
        lp_b = logprob_sum(model, tok, [stem], [bad])
        ok = (lp_g[0] > lp_b[0])
        correct += int(ok)
        if verbose:
            details.append((stem, float(lp_g[0].item()), float(lp_b[0].item()), float((lp_g - lp_b)[0].item()), ok))

    if verbose:
        print("\n[mini_morph details]")
        for p, g, b, m, ok in details:
            print(f"  {('✓' if ok else '✗')} margin={m:.2f}  good={g:.2f}  bad={b:.2f}  |  x='{p}'")

    return correct / totals

class WordBudget:
    def __init__(self, limit_words=100_000_000):
        self.limit = int(limit_words)
        self.used = 0
    def add(self, texts: List[str]):
        # rough words via whitespace
        self.used += sum(len(t.split()) for t in texts)
    def ok(self):
        return self.used <= self.limit
    

# Add helper function before generate() function (around line 300)
def pad_to_bucket(texts, tokenizer, buckets=[32, 64, 128, 256, 384, 512]):
    """
    Pad/truncate texts to fixed bucket sizes to reduce CUDAGraph recompilations.
    Returns texts padded to the nearest bucket size.
    """
    # Find max length in batch
    max_len = max(len(tokenizer.tokenize(t)) for t in texts)
    
    # Find appropriate bucket
    target_len = min((b for b in buckets if b >= max_len), default=buckets[-1])
    
    return target_len


def combined_loss(student, reference, tok, prefixes, y_star, kl_weight=0.03):
    """Compute SFT + KL in single forward pass."""
    student.train()
    
    # Encode once
    enc_x = tok(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    enc_y = tok(y_star, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    
    # Student forward (with grad)
    st_out = student(input_ids=input_ids[:, :MAX_CONCAT], 
                     attention_mask=attn_mask[:, :MAX_CONCAT],
                     labels=labels[:, :MAX_CONCAT])
    L_sft = st_out.loss
    
    # Reference forward (no grad) - reuse same inputs
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        ref_out = reference(input_ids=input_ids[:, :MAX_CONCAT], 
                           attention_mask=attn_mask[:, :MAX_CONCAT])
    
    # Efficient KL: only on y positions
    st_logits = st_out.logits[:, :-1, :]
    ref_logits = ref_out.logits[:, :-1, :]
    mask_y = (labels[:, 1:] != -100)
    
    # Use MSE on logits (faster than full KL over vocab)
    kl_approx = F.mse_loss(st_logits[mask_y], ref_logits[mask_y].detach())
    
    return L_sft + kl_weight * kl_approx, L_sft, kl_approx

import difflib
import math
from typing import List, Tuple

def _norm_logprob_per_tok(model, tok, x: str, y: str) -> float:
    lp = logprob_sum(model, tok, [x], [y])[0].item()
    ntoks = len(tok(y, add_special_tokens=False).input_ids)
    return lp / max(ntoks, 1)

def _rep_frac(text: str, n: int = 3) -> float:
    toks = text.split()
    if len(toks) < n+1: return 0.0
    seen, reps = set(), 0
    for i in range(len(toks)-n+1):
        ng = tuple(toks[i:i+n])
        reps += (ng in seen)
        seen.add(ng)
    return reps / max(len(toks)-n+1, 1)

def _prefix_leak(prefix: str, cand: str) -> float:
    # penalize if cand starts with last 3–6 words of prefix
    pw = prefix.split()[-6:]
    cw = cand.split()[:6]
    m = 0
    for k in range(6, 2, -1):
        if pw[-k:] == cw[:k]:
            m = k; break
    return float(m > 0)

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)

# @torch.no_grad()
def build_contrastive_pairs(
    student, reference, tok, prefixes: List[str], attempts: List[str],
    k_per_prefix: int = 6, margin: float = 0.4, max_len: int = 20,
    critic = None  # optional LLM critic scores
) -> Tuple[List[str], List[str], List[str]]:
    xs, chosen, rejected = [], [], []
    student.eval()
    reference.eval()
    for x, a0 in zip(prefixes, attempts):
        # pool candidates
        pool = set()
        pool.add((a0 or "").strip())
        gens = []
        # gens += complete(student, tok, [x], max_new_tokens=max_len, temperature=0.1, fast_sample=True)  # greedy
        gens += complete(student, tok, [x], max_new_tokens=max_len, temperature=0.6, top_p=0.7, fast_sample=False)
        # gens += complete(student, tok, [x], max_new_tokens=max_len, fast_sample=True)
        for g in gens:
            if g is not None:
                pool.add(g.strip())
            if len(pool) >= k_per_prefix + 2:
                break
        cand_list = [c for c in pool if c]

        if not cand_list:
            continue

        # critic-based score
        ref_scored = []
        for y in cand_list:
            s = _norm_logprob_per_tok(critic, tok, x, y)
            # s -= 0.8 * _rep_frac(y, n=3)
            # s -= 0.8 * _prefix_leak(x, y)
            # if len(y.split()) > max_len: s -= 0.5
            # if not y: s -= 1.0
            ref_scored.append((s, y))

        # optional: LLM critic reranking (e.g., caregiver-7B) for stronger contrast
        # critic(x, cands) -> list of floats (higher is better), same order as cand_list
        # critic_weight = 0.5
        # # critic = None  # disable for now
        # if critic is not None and len(cand_list) > 1:
        #     try:
        #         with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        #             crit_scores = critic(x, cand_list)  # length == len(cand_list)
        #         # blend: keep reference as anchor; critic as reranker
        #         crit_map = {y: cs for y, cs in zip(cand_list, crit_scores)}
        #         ref_scored = [ (s + critic_weight * crit_map.get(y, 0.0), y) for (s, y) in ref_scored ]
        #     except Exception:
        #         pass  # fall back to reference-only

        ref_scored.sort(key=lambda t: t[0])
        s_lo, y_lo = ref_scored[0]
        s_hi, y_hi = ref_scored[-1]

        # filter weak/near-duplicate pairs
        if (s_hi - s_lo) < margin: 
            continue
        if _jaccard(y_hi, y_lo) > 0.7: 
            continue
        if y_hi == y_lo: 
            continue
        student.train()
        xs.append(x); chosen.append(y_hi); rejected.append(y_lo)
    return xs, chosen, rejected