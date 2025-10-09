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
from peft import LoraConfig, get_peft_model

# Import the color logger
from color_logger import get_logger, MLColors, Fore, Style

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

from datasets import load_dataset, get_dataset_config_names

def force_json(text: str) -> Dict:
    # Extract first {...} block to guard against stray tokens
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in caregiver output")
    js = json.loads(m.group(0))
    return js

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
        # Use fast tokenizer for better performance
        self.tok = GPT2Tokenizer.from_pretrained('gpt2-medium', use_fast=True)
        # Set pad token if not present
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        # CRITICAL: Set padding side to left for decoder-only models
        self.tok.padding_side = 'left'
            
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')

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

    def correct_batch(self, prefixes: List[str], students: List[str]) -> List[CaregiverOutput]:
        """Batch correction for much better performance."""
        # Build batch of prompts
        batch_prompts = []
        for prefix, student in zip(prefixes, students):
            prompt = self.system["content"]
            prompt += f"Prefix: {prefix}\nStudent: {student}\nJSON: {{"
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = self.tok(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=400  # Much shorter context
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Batch generate (much faster) - Remove temperature to avoid warnings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,  # Much shorter generations
                do_sample=False,    # Greedy for speed and consistency
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id
            )
        
        # Decode batch results
        results = []
        input_length = inputs['input_ids'].shape[1]
        
        for i, (prefix, student) in enumerate(zip(prefixes, students)):
            try:
                generated_text = self.tok.decode(
                    outputs[i][input_length:], 
                    skip_special_tokens=True
                )
                full_json_text = "{" + generated_text
                
                js = force_json(full_json_text)
                out = validate_js(js)
                results.append(out)
                
            except Exception as e:
                # Fast fallback
                results.append(CaregiverOutput(
                    student.strip(), "other", None, f"Parse error: {str(e)[:30]}"
                ))
        
        return results

    def correct(self, prefix: str, student: str, shots: Optional[list]=None) -> CaregiverOutput:
        """Single correction (calls batch method for consistency)."""
        results = self.correct_batch([prefix], [student])
        return results[0]

@torch.no_grad()
def full_logprob_sum(model, tok, x_list, y_list, max_len=256):
    model.eval()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)

    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)

    # labels: predict only y tokens
    labels = input_ids.clone()
    # mask out x part
    labels[:, :enc_x.input_ids.shape[1]] = -100

    # Causal LM predicts token t at logits[:, t-1], so shift both by one step.
    # We’ll compute per-token NLL only where labels != -100.
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits

    # Shift
    logits = logits[:, :-1, :]         # predict next token
    tgt    = labels[:, 1:]             # next-token targets
    mask_y = (tgt != -100)

    # Gather logprobs for targets
    logp_all = F.log_softmax(logits, dim=-1)
    tgt_safe = tgt.masked_fill(~mask_y, 0)
    token_lp = logp_all.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)

    # Sum only over y tokens
    lp_sum = (token_lp * mask_y).sum(dim=1)
    return lp_sum


@torch.no_grad()
def eval_blimp_hf(model, tok, n_per_cat=50, max_len=64, categories=None, progress=True):
    """
    Evaluate on real BLiMP items:
      - loads each phenomenon split
      - samples up to n_per_cat test items
      - compares mean logprob of good vs bad full sentences
    Returns: overall_acc, per_category dict
    """
    if categories is None:
        categories = get_dataset_config_names("blimp")  # 67 phenomena

    per_cat = {}
    total_right, total = 0, 0

    for cat in categories:
        ds = load_dataset("blimp", cat, split="train")
        # each example has 'sentence_good' and 'sentence_bad'
        n = min(n_per_cat, len(ds))
        if n == 0:
            continue
        # sample without replacement for speed
        idx = torch.randperm(len(ds))[:n].tolist()
        good = [ds[i]["sentence_good"] for i in idx]
        bad  = [ds[i]["sentence_bad"]  for i in idx]

        # batch in chunks to save memory
        batch = 64
        rights = 0
        for s in range(0, n, batch):
            g = good[s:s+batch]
            b = bad[s:s+batch]
            lp_g = full_logprob_sum(model, tok, g, b, max_len=max_len)
            lp_b = full_logprob_sum(model, tok, b, b, max_len=max_len)
            rights += (lp_g > lp_b).sum().item()

        acc = rights / n
        per_cat[cat] = acc
        total_right += rights
        total += n
        if progress:
            print(f"[BLiMP/{cat:>28}] acc={acc:.3f}  n={n}")

    overall = total_right / max(total, 1)
    return overall, per_cat

def pretty_print_blimp(per_cat):
    # quick summary by sorting hardest → easiest
    rows = sorted(per_cat.items(), key=lambda kv: kv[1])
    print("\n[BLiMP] per-category (hardest → easiest)")
    for k, v in rows:
        print(f"  {k:>28}: {v:.3f}")


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
def generate(model, tok, prefixes, max_new_tokens=12, temperature=0.9, top_p=0.9):
    model.eval()
    ids = tok(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=480).to(DEVICE)
    
    # Remove temperature from generation call since it's causing warnings
    # Use do_sample=True with top_p for similar randomness
    outputs = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True, 
        top_p=top_p,
        pad_token_id=tok.eos_token_id, 
        eos_token_id=tok.eos_token_id
    )
    outs = []
    for i, output in enumerate(outputs):
        actual_input_len = (ids["input_ids"][i] != tok.pad_token_id).sum().item()
        gen_tokens = output[actual_input_len:]
        outs.append(tok.decode(gen_tokens, skip_special_tokens=True).strip())
    return outs


MAX_CONCAT = 512  # enough for short prefixes + short corrections

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

@torch.no_grad()
def logprob_sum(model, tok, x_list, y_list, max_len=256):
    model.eval()
    enc_x = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)
    enc_y = tok(y_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len//2).to(DEVICE)

    input_ids = torch.cat([enc_x.input_ids, enc_y.input_ids], dim=1)
    attn_mask = torch.cat([enc_x.attention_mask, enc_y.attention_mask], dim=1)

    # labels: predict only y tokens
    labels = input_ids.clone()
    # mask out x part
    labels[:, :enc_x.input_ids.shape[1]] = -100

    # Causal LM predicts token t at logits[:, t-1], so shift both by one step.
    # We’ll compute per-token NLL only where labels != -100.
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits

    # Shift
    logits = logits[:, :-1, :]         # predict next token
    tgt    = labels[:, 1:]             # next-token targets
    mask_y = (tgt != -100)

    # Gather logprobs for targets
    logp_all = F.log_softmax(logits, dim=-1)
    tgt_safe = tgt.masked_fill(~mask_y, 0)
    token_lp = logp_all.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)

    # Sum only over y tokens
    lp_sum = (token_lp * mask_y).sum(dim=1)
    return lp_sum / len(tok)


def kl_to_ref(student, reference, tok, x_list):
    student.train(); reference.eval()
    enc = tok(x_list, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
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
    with torch.no_grad():
        margin_ref = (lr_pos - lr_neg)
    margin_student = (lp_pos - lp_neg)
    print(f"[DPO] mean margins | student: {margin_student.mean().item():.3f}  ref: {margin_ref.mean().item():.3f}")
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
    ("The key to the cabinets", " is missing.", " are missing."),
    ("Each of the boys", " is late.", " are late."),
]

MORPH_STEMS   = ["She showed much", "His"]
MORPH_GOOD    = [" creativity.", " activity impressed them."]
MORPH_BAD     = [" creativeness.", " activeness impressed them."]

@torch.no_grad()
def simple_logprob(model, tok, full_text):
    """Calculate logprob of full text sequence (simpler than logprob_sum)."""
    model.eval()
    # Temporarily switch to right padding for evaluation to avoid complexity
    # orig_padding_side = tok.padding_side
    # tok.padding_side = 'right'
    
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
def mini_blimp(model, tok, verbose=False):
    """
    Compare logprob of the continuation only (y) conditioned on the same prefix (x).
    Uses padding-agnostic logprob_sum to avoid any left/right padding mismatch.
    """
    model.eval()
    correct = 0
    totals = len(BLIMP_MINI)
    details = []

    for prefix, good, bad in BLIMP_MINI:
        lp_g = logprob_sum(model, tok, [prefix], [good])  # log P(y_good | x)
        lp_b = logprob_sum(model, tok, [prefix], [bad])   # log P(y_bad  | x)
        ok = (lp_g[0] > lp_b[0])
        correct += int(ok)
        if verbose:
            details.append((prefix, float(lp_g[0].item()), float(lp_b[0].item()), float((lp_g - lp_b)[0].item()), ok))

    if verbose:
        print("\n[mini_blimp details]")
        for p, g, b, m, ok in details:
            print(f"  {('✓' if ok else '✗')} margin={m:.2f}  good={g:.2f}  bad={b:.2f}  |  x='{p}'")

    return correct / totals

@torch.no_grad()
def mini_morph(model, tok, verbose=False):
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
    # Use fast tokenizer explicitly for better performance
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Fix padding side for decoder-only models (critical for generation quality)
    tok.padding_side = 'left'  # This should be the ONLY place we set padding_side
    logger.info(f"Set tokenizer padding_side to: {Fore.GREEN}left{Style.RESET_ALL} (required for decoder-only models)")
    logger.info(f"Using fast tokenizer: {Fore.GREEN}{tok.is_fast}{Style.RESET_ALL}")
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
    caregiver = Caregiver()

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

    logger.info("Running initial evaluation on REAL BLiMP…")
    # bl0, per_cat0 = eval_blimp_hf(student, tok, n_per_cat=5, max_len=64, progress=True)
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

        # Light nudge: append a small cue token to expose the locus (helps heuristics)
        prefixes = [p + " is" if re.search(r"\b\w+s\b$", p) else p for p in prefixes]
        
        # Generate short student attempts
        gen_start = time.time()
        attempts = generate(student, tok, prefixes, max_new_tokens=12, temperature=0.9, top_p=0.9)
        gen_time = time.time() - gen_start
        step_times["generate"] += gen_time

        # Caregiver corrections
        correct_start = time.time()
        try:
            # Use batch correction for much better performance
            outs = caregiver.correct_batch(prefixes, attempts)
            y_star = [o.corrected for o in outs]
            correct_time = time.time() - correct_start
            step_times["correct"] += correct_time
        except Exception as e:
            # Fallback: use original attempts as corrections
            outs = [CaregiverOutput(attempt, "other", None, "Caregiver failed") for attempt in attempts]
            y_star = attempts
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

        # --- Compute core losses first ---
        forward_start = time.time()
        L_sft = ce_targets(student, tok, prefixes, y_star)
        L_kl  = kl_to_ref(student, reference, tok, prefixes)
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
                if (lp_pos_ref - lp_neg_ref).item() < 0.5:
                    continue
                xs_dpo.append(x)
                yp.append(o.corrected)
                yn.append(o.negative)

            if xs_dpo:
                L_dpo = dpo_loss(student, reference, tok, xs_dpo, yp, yn, beta=2.0)

        step_times["forward"] += time.time() - forward_start

        # --- Combine losses (final total) ---
        dpo_weight = 1.0 if args.use_dpo else 0.0
        loss = L_sft + 0.03 * L_kl + dpo_weight * L_dpo



        # weight DPO more heavily once we filter for high-confidence pairs
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
    final_bl = mini_blimp(student, tok)
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
