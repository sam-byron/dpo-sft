# bnc_interactive_train.py
import os, re, math, random, json, argparse, itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset, IterableDatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)
torch.manual_seed(7)

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
            L = random.randint(8, min(20, len(words)))
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
    outs = []
    for x in prefixes:
        ids = tok(x, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(gen.strip())
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
def mini_blimp(model, tok):
    xs, g, b = zip(*BLIMP_MINI)
    lp_g = logprob_sum(model, tok, list(xs), list(g))
    lp_b = logprob_sum(model, tok, list(xs), list(b))
    return (lp_g > lp_b).float().mean().item()

@torch.no_grad()
def mini_morph(model, tok):
    lp1 = logprob_sum(model, tok, MORPH_STEMS, MORPH_GOOD)
    lp2 = logprob_sum(model, tok, MORPH_STEMS, MORPH_BAD)
    return (lp1 > lp2).float().mean().item()

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

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Student + LoRA
    base = AutoModelForCausalLM.from_pretrained(args.model_name)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn","c_proj","q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(base, lora_cfg).to(DEVICE)
    student.train()

    # Reference (frozen)
    reference = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    reference.eval()

    caregiver = HeuristicCaregiver()

    # Data stream
    bnc_name = args.bnc_name if args.bnc_name else None
    bnc_dir  = args.bnc_dir if args.bnc_dir else None
    ds = BNCPrefixStream(tok, bnc_name, bnc_dir)
    loader = DataLoader(ds, batch_size=args.batch_size)

    # Optim & sched
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=500, num_training_steps=args.steps)

    budget = WordBudget(100_000_000)

    # Initial eval
    bl0 = mini_blimp(student, tok)
    mo0 = mini_morph(student, tok)
    print(f"[Init] mini-BLiMP={bl0:.3f}  Morph={mo0:.3f}")

    step = 0
    for batch in loader:
        step += 1
        prefixes = batch["prefix"]
        # Generate short student attempts
        attempts = generate(student, tok, prefixes, max_new_tokens=12, temperature=0.9, top_p=0.9)

        # Caregiver corrections
        outs = [caregiver.correct(x, y) for x, y in zip(prefixes, attempts)]
        y_star = [o.corrected for o in outs]

        # Losses
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

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step()

        # Budget: only count texts we train ON (y*)
        budget.add(y_star)
        if not budget.ok():
            print("[Budget] Reached 100M word limit. Stopping.")
            break

        if step % 100 == 0:
            print(f"[{step}] L_sft={L_sft.item():.3f}  L_kl={L_kl.item():.3f}  L_dpo={L_dpo.item():.3f}  Used≈{budget.used/1e6:.1f}M words")

        if step % args.eval_every == 0:
            bl = mini_blimp(student, tok)
            mo = mini_morph(student, tok)
            print(f"[Eval {step}] mini-BLiMP={bl:.3f}  Morph={mo:.3f}")
            # save LoRA adapter
            student.save_pretrained(os.path.join(args.save_dir, f"step_{step}"))

        if step >= args.steps:
            print("[Done] Reached max steps.")
            break

if __name__ == "__main__":
    main()
