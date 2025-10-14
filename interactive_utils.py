import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, cast
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
from transformers import AutoTokenizer, AutoModelForCausalLM
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

@torch.no_grad()
def get_uncertainty(student, tok, prefixes, attempts):
    """Return per-sample uncertainty (entropy) as selection criterion."""
    student.eval()
    enc_x = tok(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    enc_y = tok(attempts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    # Ensure input_ids and attention_mask are long tensors before moving to device
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long().to(DEVICE)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long().to(DEVICE)
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        out = student(input_ids=input_ids[:, :MAX_CONCAT], attention_mask=attn_mask[:, :MAX_CONCAT])
    
    logits = out.logits[:, enc_x.input_ids.shape[1]-1:-1, :]
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=-1)
    return entropy.cpu().tolist()

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
    with torch.no_grad():
        # Reference preferences (detach)
        lp_pos_ref = logprob_sum(reference, tok, xs, y_pos, max_len=max_len)  # [B]
        lp_neg_ref = logprob_sum(reference, tok, xs, y_neg, max_len=max_len)  # [B]
        ref_delta = lp_pos_ref - lp_neg_ref                                  # [B]

    # Student preferences (requires grad)
    lp_pos_stu = logprob_sum(student, tok, xs, y_pos, max_len=max_len)       # [B]
    lp_neg_stu = logprob_sum(student, tok, xs, y_neg, max_len=max_len)       # [B]
    stu_delta  = lp_pos_stu - lp_neg_stu

    # Score and logistic loss
    margin = beta * (stu_delta - ref_delta)
    # log σ(z) = -softplus(-z)  ; we want -E[log σ(margin)]
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


class Caregiver:
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct', rng_seed: int = 0):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'
            
        # Load in bfloat16 directly
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')  # type: ignore[call-arg]       

        # Stronger system prompt with constraints + inline few-shots
        self.system_prompt = (
            "You are a precise suffix corrector. You receive a sentence in two parts: Prefix and Suffix. "
            "Correct ONLY the Suffix so that 'Prefix + Suffix' is grammatical, coherent, and faithful in meaning. "
            "Make the minimum necessary edits (grammar, agreement, semantics, punctuation). Do not change the Prefix. "
            "If the Suffix is already acceptable, return it unchanged. If Suffix is empty, return the shortest natural continuation. "
            "Output ONLY the corrected Suffix. No quotes, no labels, no explanations. Keep it concise (<= 15 tokens). "
            "Do not repeat the Prefix.\n\n"
            "Examples:\n"
            "Prefix: The cat sat\nSuffix: in the sun.\nAnswer: in the sun.\n"
            "Prefix: She\nSuffix: see themselves.\nAnswer: saw herself.\n"
            "Prefix: They\nSuffix: is happy.\nAnswer: are happy.\n"
            "Prefix: Judge Stanley Spence told Creagh: 'You were a\nSuffix:\nAnswer: good man.'\n"
            "Prefix: In a business any operating loss\nSuffix: has been carried down\nAnswer: has been carried forward\n"
            "Prefix: He pointed ahead through\nSuffix: the room to be taken on\nAnswer: the door\n"
        )

        # Few-shot pairs used as chat turns (helps small models)
        self._few_shots = [
            ("The cat sat", "in the sun.", "in the sun."),
            ("She", "see themselves.", "saw herself."),
            ("They", "is happy.", "are happy."),
            ("Judge Stanley Spence told Creagh: 'You were a", "", "good man.'"),
            ("In a business any operating loss", "has been carried down", "has been carried forward"),
            ("He pointed ahead through", "the room to be taken on", "the door"),
        ]

    # --- helpers ---
    def _build_messages(self, prefix: str, suffix: str):
        msgs = [{"role": "system", "content": self.system_prompt}]
        for p, s, a in self._few_shots:
            msgs.append({"role": "user", "content": f"Prefix: {p}\nSuffix: {s}\nReturn only the corrected Suffix."})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": f"Prefix: {prefix}\nSuffix: {suffix}\nReturn only the corrected Suffix."})
        return msgs

    def _clean_suffix(self, text: str, forbid_prefixes: List[str]) -> str:
        t = text.strip()
        # Cut at first newline or chat marker
        cut_markers = ["\nPrefix:", "\nSuffix:", "\nSystem:", "\nUser:", "\nAssistant:"]
        for m in cut_markers:
            idx = t.find(m)
            if idx != -1:
                t = t[:idx]
        # Strip wrapping quotes
        if len(t) >= 2 and ((t[0], t[-1]) in {('"', '"'), ("'", "'")}):
            t = t[1:-1].strip()
        # Remove any leaked labels
        t = re.sub(r'^\s*(Prefix:|Suffix:|Answer:)\s*', '', t, flags=re.I)
        # If model echoed any forbidden label, drop everything before last colon
        for fp in forbid_prefixes:
            pos = t.find(fp)
            if pos != -1:
                t = t[:pos]
        # Collapse spaces
        t = " ".join(t.split())
        # Cap to 15 tokens
        toks = t.split()
        if len(toks) > 15:
            t = " ".join(toks[:15])
        return t

    def correct_batch(self, prefixes: List[str], students: List[str]) -> List[CaregiverOutput]:
        prompts = [self._build_messages(p, s) for p, s in zip(prefixes, students)]

        # Build chat inputs per sample (with few-shots included)
        self.model.eval()
        self.tok.padding_side = "left"

        chat_payloads = self.tok.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = self.tok(
            chat_payloads,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(DEVICE)

        # Disallow label words to reduce leakage
        bad_words = ["Prefix:", "Suffix:", "System:", "User:", "Assistant:", "Answer:"]
        bad_words_ids = [self.tok.encode(w, add_special_tokens=False) for w in bad_words if w]

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
            out_ids = self.model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=20,
                do_sample=False,              # greedy for stability on 1.5B
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_ids if bad_words_ids else None,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                use_cache=True,
            )

        # Keep only generated tail
        gen_only = [o[len(i):] for i, o in zip(enc.input_ids, out_ids)]
        texts = self.tok.batch_decode(gen_only, skip_special_tokens=True)

        cleaned = [self._clean_suffix(t, forbid_prefixes=["\nPrefix:", "\nSuffix:", "Answer:"]) for t in texts]
        return [CaregiverOutput(c) for c in cleaned]

    def correct(self, prefix: str, student: str) -> CaregiverOutput:
        """Single correction."""
        results = self.correct_batch([prefix], [student])
        return results[0]

@torch.no_grad()
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

@torch.no_grad()
def generate(model, tok, prefixes, max_new_tokens=12, temperature=1.0, top_p=0.9):
    model.eval()
    # lengths = [len(p.split()) for p in prefixes]
    # order = sorted(range(len(prefixes)), key=lambda i: lengths[i])
    # rev = [0]*len(order)
    # for i, oi in enumerate(order): rev[oi] = i
    # ordered = [prefixes[i] for i in order]
    target_len = pad_to_bucket(prefixes, tok)
    enc = tok(prefixes, return_tensors="pt", padding="max_length",
              max_length=target_len, truncation=True).to(DEVICE)
    
    # CRITICAL FIX: Ensure attention_mask is set correctly for left-padded sequences
    # Left padding means pad tokens are at the START, so we need to shift the mask
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # Debugging: check if left padding is causing issues
    # Print first example to verify padding
    if torch.rand(1).item() < 0.01:  # 1% of batches
        print(f"[DEBUG] First input_ids: {input_ids[0][:20].tolist()}")
        print(f"[DEBUG] First attn_mask: {attention_mask[0][:20].tolist()}")
        print(f"[DEBUG] Decoded prefix: {tok.decode(input_ids[0][attention_mask[0].bool()], skip_special_tokens=False)}")
    
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,  # Penalize repeating tokens
            no_repeat_ngram_size=3,  # Prevent exact 3-gram repeats
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )

    out = [
        output_ids[len(input_ids_):] for input_ids_, output_ids in zip(input_ids, out)
    ]
    return [tok.decode(g, skip_special_tokens=True).strip() for g in out]
    # Calculate actual input lengths (non-padding tokens)
    # input_lens = attention_mask.sum(dim=1).tolist()
    # gens = []
    # for i, seq in enumerate(out):
    #     gen_part = seq[input_lens[i]:]
    #     gens.append(tok.decode(gen_part, skip_special_tokens=True).strip())
    # restored = [None]*len(gens)
    # for i, gi in enumerate(gens):
    #     restored[order[i]] = gi
    # return restored

def ce_targets(student, tok, x_list, y_list):
    student.train()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2)
    # Ensure input_ids and attention_mask are long tensors
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).to(DEVICE).long()
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).to(DEVICE).long()
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    out = student(input_ids=input_ids[:, :MAX_CONCAT],
                  attention_mask=attn_mask[:, :MAX_CONCAT],
                  labels=labels[:, :MAX_CONCAT])
    return out.loss

# logprob_sum: remove incorrect divide by len(tok) and use autocast
@torch.no_grad()
def logprob_sum(model, tok, x_list, y_list, max_len=256):
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
        enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
        
        # Ensure input_ids are Long (not Float) - critical fix!
        input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long()
        attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long()
        
        labels = input_ids.clone()
        labels[:, :enc_x.input_ids.shape[1]] = -100
        
        out = model(input_ids=input_ids, attention_mask=attn_mask)
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
    Student needs gradients, reference doesn't.
    """
    student.train()  # ✅ Keep student in train mode for gradients
    reference.eval()
    
    # Encode x and y (only positives by default)
    enc_x = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    enc_y = tok(y_pos, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    
    # Build inputs and masks - ensure Long dtype
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1).long()
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1).long()
    
    # Labels to identify y positions (mask x with -100)
    labels = input_ids.clone()
    labels[:, :enc_x.input_ids.shape[1]] = -100
    
    # Student forward WITH gradients (no torch.no_grad!)
    st_out = student(input_ids=input_ids[:, :MAX_CONCAT], attention_mask=attn_mask[:, :MAX_CONCAT])
    
    # Reference forward WITHOUT gradients
    with torch.no_grad():
        rf_out = reference(input_ids=input_ids[:, :MAX_CONCAT], attention_mask=attn_mask[:, :MAX_CONCAT])
    
    # Shift for next-token prediction
    st_logits = st_out.logits[:, :-1, :] / temperature
    rf_logits = rf_out.logits[:, :-1, :] / temperature
    mask_y = (labels[:, 1:] != -100)  # positions belonging to y

    # KL(student || reference) over vocab at each token
    log_p = F.log_softmax(st_logits, dim=-1)
    q = F.softmax(rf_logits, dim=-1)

    # Per-token KL: sum over vocab, then mask to y positions
    kl_tok = F.kl_div(log_p, q, reduction="none").sum(dim=-1)  # [B, T]
    kl_tok = kl_tok * mask_y.float()
    denom = mask_y.float().sum().clamp_min(1.0)
    return kl_tok.sum() / denom

@torch.no_grad()
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
    with torch.no_grad():
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

def build_contrastive_pairs(
    student, reference, tok, prefixes: List[str], attempts: List[str],
    k_per_prefix: int = 6, margin: float = 0.4, max_len: int = 15
) -> Tuple[List[str], List[str], List[str]]:
    xs, chosen, rejected = [], [], []
    for x, a0 in zip(prefixes, attempts):
        # pool candidates
        pool = set()
        pool.add(a0.strip())
        gens = []
        gens += generate(student, tok, [x], max_new_tokens=max_len, temperature=0.0, top_p=1.0)  # greedy
        gens += generate(student, tok, [x], max_new_tokens=max_len, temperature=0.7, top_p=0.9)
        gens += generate(student, tok, [x], max_new_tokens=max_len, temperature=0.9, top_p=0.95)
        for g in gens:
            pool.add((g or "").strip())
            if len(pool) >= k_per_prefix+2: break
        cand_list = [c for c in pool if c]

        # score
        scored = []
        for y in cand_list:
            s = _norm_logprob_per_tok(reference, tok, x, y)
            s -= 0.8 * _rep_frac(y, n=3)
            s -= 0.8 * _prefix_leak(x, y)
            if len(y.split()) > max_len: s -= 0.5
            if not y: s -= 1.0
            scored.append((s, y))

        if not scored: continue
        scored.sort(key=lambda t: t[0])
        s_lo, y_lo = scored[0]
        s_hi, y_hi = scored[-1]

        # filter weak/near-duplicate pairs
        if (s_hi - s_lo) < margin: continue
        if _jaccard(y_hi, y_lo) > 0.7: continue
        if y_hi == y_lo: continue

        xs.append(x); chosen.append(y_hi); rejected.append(y_lo)
    return xs, chosen, rejected