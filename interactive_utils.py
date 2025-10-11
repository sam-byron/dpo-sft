
import os, re, math, random, json, argparse, itertools, time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, cast

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
    enc_x = tok(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    enc_y = tok(attempts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)
    
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
    tag: str
    negative: Optional[str]
    reason: str


ALLOWED_TAGS = {"agreement","reflexive","npi","entity","morphology","other"}

def validate_js(js: Dict) -> CaregiverOutput:
    for k in ["corrected","tag","negative","reason"]:
        if k not in js: raise ValueError(f"Missing key: {k}")
    tag = js["tag"].strip().lower()
    if tag not in ALLOWED_TAGS: tag = "other"
    corrected = js["corrected"].strip()
    negative  = js["negative"]
    if isinstance(negative, str):
        negative = negative.strip()
        if not negative: negative = None
    elif negative is not None:
        negative = None
    reason = js["reason"].strip()
    # Basic sanity checks
    if len(corrected.split()) > 24:  # small slack
        corrected = " ".join(corrected.split()[:24])
    if negative and len(negative.split()) > 24:
        negative = " ".join(negative.split()[:24])
    return CaregiverOutput(corrected, tag, negative, reason)

class Caregiver:
    """Very small heuristic caregiver. You can later swap to an LLM-based one."""
    def __init__(self, rng_seed: int = 0):
        self.tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'
            
        # Load in bfloat16 directly (saves memory + faster)
        self.model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-1.5B-Instruct',
            torch_dtype=torch.bfloat16  # Native bf16 weights
        )
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')  # type: ignore[call-arg]

        self.system = {
            "role": "system",
            "content": (
                "You are a precise language teacher. "
                "Given a student's short output y (≤20 tokens) for a prefix x, you must:\n"
                "1) Provide a corrected sentence y* that preserves meaning while fixing grammar/morphology/reference issues.\n"
                "2) Assign exactly one tag from: agreement, reflexive, npi, entity, morphology, other.\n"
                "3) Optionally provide a minimal negative y− differing from y* by ONE targeted error; else null.\n"
                "4) Return STRICT JSON matching the Schema. Do not include code fences or explanations.\n"
                "Constraints: concise y* (≤20 tokens); minimal edits."
                "Schema: {\"corrected\": \"string\", \"tag\": \"agreement|reflexive|npi|entity|morphology|other\", \"negative\": \"string|null\", \"reason\": \"string\"}"
            )
        }
        self.shots = [
            {"corrected": "The keys to the cabinet are on the table.", "tag": "agreement", "negative": "The keys to the cabinet is on the table.", "reason": "Plural subject requires 'are'."},
            {"corrected": "John told Mary that he will go.", "tag": "entity", "negative": "John told Mary that she will go.", "reason": "Pronoun must refer to John."},
            {"corrected": "No student has ever cheated.", "tag": "npi", "negative": "A student has ever cheated.", "reason": "'ever' needs a negative licensor."}
        ]

        self.system_text = self.system["content"]
        # Pre-tokenize system prompt once (CPU tensor retained; will be moved on use)
        self.system_ids = self.tok(self.system_text, return_tensors="pt", add_special_tokens=False)

    # Optimize correct_batch to reuse pre-tokenized system and use autocast/inference_mode
    def correct_batch(self, prefixes: List[str], students: List[str]) -> List[CaregiverOutput]:
        batch_snippets = [
            f"Prefix: {p}\nStudent: {s}\nJSON: {{"
            for p, s in zip(prefixes, students)
        ]
        
        # Use bucketed padding for stable shapes
        target_len = pad_to_bucket(batch_snippets, self.tok, buckets=[64, 128, 256])
        dyn = self.tok(batch_snippets, return_tensors="pt", padding='max_length', 
                      max_length=target_len, truncation=True, add_special_tokens=False)
        
        # Concatenate system_ids + dynamic part
        input_ids = torch.cat([self.system_ids["input_ids"].expand(dyn["input_ids"].size(0), -1), dyn["input_ids"]], dim=1)
        attn_mask = torch.cat([torch.ones_like(self.system_ids["input_ids"]).expand(dyn["attention_mask"].size(0), -1), dyn["attention_mask"]], dim=1)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=48,
                do_sample=True,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
            )

        # Slice only newly generated tokens
        new_tokens = gen[:, input_ids.size(1):]
        results = []
        for i, (pfx, stu) in enumerate(zip(prefixes, students)):
            try:
                txt = self.tok.decode(new_tokens[i], skip_special_tokens=True)
                js = force_json("{" + txt)
                results.append(validate_js(js))
            except Exception as e:
                results.append(CaregiverOutput(stu.strip(), "other", None, f"Parse error: {str(e)[:24]}"))
        return results

    def correct(self, prefix: str, student: str, shots: Optional[list]=None) -> CaregiverOutput:
        """Single correction (calls batch method for consistency)."""
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

class BNCPrefixStream(Dataset):
    """
    Map-style dataset of short prefixes (8–20 tokens) from BNC-like text.
    If bnc_name is provided, loads HF dataset (non-streaming) and materializes items;
    otherwise reads *.txt files under bnc_dir. Each item is a dict {"prefix", "tag"}.
    """
    def __init__(
        self,
        tokenizer,
        bnc_name: Optional[str],
        bnc_dir: Optional[str],
        max_len_tokens: int = 20,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.tok = tokenizer
        self.bnc_name = bnc_name
        self.bnc_dir = bnc_dir
        self.max_len = max_len_tokens
        self.limit = limit
        self.rng = random.Random(seed)
        self.items: List[Dict[str, str]] = []

        if self.bnc_name:
            self._collect_hf()
        elif self.bnc_dir:
            self._collect_dir()
        else:
            self.items = []

    def _maybe_add_sentence(self, sent: str):
        sent = clean_text(sent)
        if not sent:
            return
        toks = self.tok.tokenize(sent)
        if not (4 <= len(toks) <= 48):
            return
        words = sent.split()
        if len(words) < 4:
            return
        min_len = 4
        max_len = min(self.max_len, len(words))
        if max_len < min_len:
            return
        # Prefer longer prefixes when possible
        if max_len >= 8:
            L = self.rng.randint(8, max_len)
        else:
            L = self.rng.randint(min_len, max_len)
        prefix = " ".join(words[:L])
        tag = "other"
        for k, rgx in PHENOMENON_PATTERNS.items():
            if rgx.match(prefix):
                tag = k
                break
        self.items.append({"prefix": prefix, "tag": tag})

    def _collect_hf(self):
        # self.bnc_name is guaranteed non-None when this is called
        name = cast(str, self.bnc_name)
        ds = load_dataset(name, split="train")  # materialized dataset
        for ex in ds:
            text = (ex.get("text") if isinstance(ex, dict) else None) or (
                ex.get("content") if isinstance(ex, dict) else None
            ) or ""
            if not text:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", text):
                self._maybe_add_sentence(sent)
                if self.limit is not None and len(self.items) >= self.limit:
                    return

    def _collect_dir(self):
        # self.bnc_dir is guaranteed non-None when this is called
        dir_path = cast(str, self.bnc_dir)
        for root, _, files in os.walk(dir_path):
            for f in files:
                if not f.lower().endswith(".txt"):
                    continue
                with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        for sent in re.split(r"(?<=[.!?])\s+", clean_text(line)):
                            self._maybe_add_sentence(sent)
                            if self.limit is not None and len(self.items) >= self.limit:
                                return

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


def batchify(items, bs):
    it = iter(items)
    while True:
        b = list(itertools.islice(it, bs))
        if not b: break
        yield b

@torch.no_grad()
def generate(model, tok, prefixes, max_new_tokens=12, temperature=0.9, top_p=0.9):
    model.eval()
    # Sort by length to reduce padding (then unsort)
    lengths = [len(p.split()) for p in prefixes]
    order = sorted(range(len(prefixes)), key=lambda i: lengths[i])
    rev = [0]*len(order)
    for i, oi in enumerate(order):
        rev[oi] = i
    ordered = [prefixes[i] for i in order]

    # Use bucketed padding for stable shapes
    target_len = pad_to_bucket(ordered, tok)
    enc = tok(ordered, return_tensors="pt", padding='max_length', 
              max_length=target_len, truncation=True).to(DEVICE)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_available() else None):
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
    # Compute actual lengths once
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    gens = []
    for i, seq in enumerate(out):
        gen_part = seq[input_lens[i]:]
        gens.append(tok.decode(gen_part, skip_special_tokens=True).strip())

    # Unsort
    restored = [None]*len(gens)
    for i, gi in enumerate(gens):
        restored[order[i]] = gi
    return restored

def ce_targets(student, tok, x_list, y_list):
    student.train()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONCAT//2).to(DEVICE)
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)
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
    
    # Build inputs and masks
    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)
    
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