#!/usr/bin/env python3
"""
blimp_sanity.py — Pseudo log-likelihood (PLL) sanity check on BLiMP and BLiMP-supplement

Overview
        Evaluates a masked language model (BERT-style) using pseudo log-likelihood:
        mask one position at a time and sum the log-prob of the original token. The
        script supports Hugging Face BLiMP subsets and an optional BLiMP-supplement
        JSONL corpus with sentence_good/sentence_bad pairs.

Usage
        python blimp_sanity.py --model_path <checkpoint_dir> \
                [--subset SUBJECT_VERB_AGREEMENT] [--max_examples 200] \
                [--device cuda] [--split auto|train|test|validation] \
                [--normalize none|per_token] [--dump 5] \
                [--benchmark blimp|blimp_supplement|both] \
                [--supplement_dir /path/to/jsonl] [--supplement_task task_name]

Arguments (selected)
        --model_path            Path to a directory with config.json + weights (and tokenizer)
        --subset                Single BLiMP subset; if omitted, runs all available
        --max_examples          Cap examples per subset/task for speed
        --normalize             "per_token" divides total PLL by token count (length norm)
        --dump                  Save worst-K examples by gap for diagnostics
        --benchmark             Which benchmark(s) to run: BLiMP, supplement, or both
        --supplement_dir        Folder containing <task>.jsonl files for supplement
        --supplement_task       Only run this supplement task name (filename stem)

Innovations & efficiency
        - Vocab clamp: if tokenizer vocab > model vocab, IDs ≥ model.vocab_size are mapped
            to [UNK] to avoid index errors when tokenizers/models mismatch.
        - Two loading paths: tries AutoModelForMaskedLM; falls back to manual import of the
            saved modeling/config files for custom architectures.
        - Per-token normalization option reports PLL per token for length-robust comparisons.
        - Worst-K example dump shows token-level contributors for quick error analysis.
"""
import argparse, math, os, csv, json
from typing import List, Tuple, Dict, Any, Optional
import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def pick_split(dataset_name: str, subset: str, split_arg: str):
    """Pick a valid split for a HF dataset subset.

    If split_arg != "auto", returns it verbatim; else probes validation, test, then train.
    Raises if none are available.
    """
    if split_arg != "auto":
        return split_arg
    for sp in ("validation", "test", "train"):
        try:
            _ = load_dataset(dataset_name, subset, split=sp)
            return sp
        except Exception:
            continue
    raise RuntimeError(f"No split found for {dataset_name}/{subset} (tried validation/test/train)")

def ar_score_with_tokens(model, tok, text: str, device: torch.device, normalize: str = "none"):
    """Autoregressive log-likelihood for causal LMs (e.g., GPT-2).

    Sum over positions i=1..L-1 of log p(x_i | x_<i). Optionally normalize per token.
    Returns (total, contribs, enc) where contribs are (pos, token_id, logp, token_str).
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]  # [1, L]
    # Safety: clamp IDs exceeding model vocab (rare but safer across mismatched tokenizers)
    vocab_limit = int(getattr(model.config, "vocab_size", input_ids.max().item() + 1))
    if input_ids.max().item() >= vocab_limit:
        # GPT-2 has no unk; fall back to eos if needed
        unk_like = tok.unk_token_id
        if unk_like is None:
            unk_like = getattr(tok, "eos_token_id", 0)
        input_ids = input_ids.clamp_max(vocab_limit - 1)
        enc["input_ids"] = input_ids
    input_ids = input_ids.to(device).long()
    attn = torch.ones_like(input_ids, device=device)
    L = input_ids.size(1)

    contribs: List[Tuple[int, int, float, str]] = []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=(device.type == "cuda")):
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [1, L, V]
        if L <= 1:
            return 0.0, [], enc
        logp_all = torch.log_softmax(logits[:, :-1, :], dim=-1)   # predict token t given <t
        target = input_ids[:, 1:]                                  # [1, L-1]
        token_lp = logp_all.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        steps = int(token_lp.numel())
        total = float(token_lp.sum().item())

    if normalize == "per_token" and steps > 0:
        total = total / steps
    # Build contribs for minimal parity with PLL output
    ids = target[0].tolist()
    vals = token_lp[0].tolist()
    offset = 1  # positions correspond to target positions (shifted by 1)
    for i, (tid, lp) in enumerate(zip(ids, vals), start=offset):
        tok_str = tok.convert_ids_to_tokens([tid])[0]
        contribs.append((i, tid, float(lp), tok_str))
    return total, contribs, enc

def find_blimp_supplement_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Try to locate the BLiMP supplement JSONL folder.
    Preference order: explicit -> ./blimp_supplement -> ../evaluation-pipeline-2024-fresh/evaluation_data/supplement_filtered
    -> ./evaluation-pipeline-2024-fresh/evaluation_data/supplement_filtered
    """
    """Return a directory path containing BLiMP-supplement JSONL files.

    Accepts an explicit path (dir or file in dir). Otherwise searches common
    local locations near the repository root.
    """
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        # allow file path pointing to a jsonl file; use its dir
        if os.path.isfile(explicit):
            return os.path.dirname(os.path.abspath(explicit))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "blimp_supplement"),
        os.path.normpath(os.path.join(here, "..", "evaluation-pipeline-2024-fresh", "evaluation_data", "supplement_filtered")),
        os.path.join(here, "evaluation-pipeline-2024-fresh", "evaluation_data", "supplement_filtered"),
        os.path.join(os.getcwd(), "blimp_supplement"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None

def get_available_supplement_tasks(supp_dir: str) -> List[str]:
    """List available task names (filename stems) from a supplement directory."""
    tasks: List[str] = []
    try:
        for fn in os.listdir(supp_dir):
            if fn.endswith(".jsonl"):
                tasks.append(os.path.splitext(fn)[0])
    except Exception:
        pass
    return sorted(tasks)

def run_supplement_subset(
    model,
    tok,
    task_name: str,
    jsonl_path: str,
    device: torch.device,
    limit: int,
    normalize: str,
    dump: int = 0,
):
    """Run PLL evaluation on a supplement JSONL file with sentence_good/sentence_bad pairs.

    Returns a dict with fields: task, n, acc, oov_rate, normalize, examples, source
    where acc is the proportion of pairs with PLL(good) > PLL(bad).
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Supplement file not found: {jsonl_path}")
    rows_raw: List[Dict[str, Any]] = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # must contain sentence_good/sentence_bad
                if "sentence_good" in obj and "sentence_bad" in obj:
                    rows_raw.append(obj)
            except Exception:
                continue
            if limit > 0 and len(rows_raw) >= limit:
                break

    wins = 0
    total = 0
    oov_tokens = 0
    total_tokens = 0
    gaps: List[Tuple[float, Dict[str, Any]]] = []

    for row in tqdm(rows_raw, desc=f"supplement:{task_name}"):
        good = row["sentence_good"]
        bad = row["sentence_bad"]
        s_good, contrib_g, enc_g = ar_score_with_tokens(model, tok, good, device, normalize=normalize)
        s_bad, contrib_b, enc_b = ar_score_with_tokens(model, tok, bad, device, normalize=normalize)
        gap = s_good - s_bad
        wins += int(gap > 0.0)
        total += 1
        # OOV stats
        unk = tok.unk_token_id
        if unk is not None and unk >= 0:
            oov_tokens += int((enc_g["input_ids"] == unk).sum().item() + (enc_b["input_ids"] == unk).sum().item())
        total_tokens += int(enc_g["input_ids"].numel() + enc_b["input_ids"].numel())
        if dump > 0:
            gaps.append((gap, {
                "good": good,
                "bad": bad,
                "gap": gap,
                "s_good": s_good,
                "s_bad": s_bad,
                "good_top_neg_tokens": sorted(contrib_g, key=lambda t: t[2])[:3],
                "bad_top_pos_tokens": sorted(contrib_b, key=lambda t: -t[2])[:3],
            }))

    acc = (wins / max(1, total)) if total > 0 else 0.0
    oov_rate = oov_tokens / max(1, total_tokens)
    worst: List[Dict[str, Any]] = []
    if dump > 0 and gaps:
        worst = [info for _, info in sorted(gaps, key=lambda x: x[0])[:dump]]
    return {
        "task": task_name,
        "n": total,
        "acc": acc,
        "oov_rate": oov_rate,
        "normalize": normalize,
        "examples": worst,
        "source": os.path.basename(jsonl_path),
    }

def ensure_subsets_list(dataset_name: str) -> List[str]:
    """Return BLiMP subset names, falling back to a static list if necessary."""
    try:
        return get_dataset_config_names(dataset_name)
    except Exception:
        # fallback to a static list (common BLiMP subsets)
        return [
            "adjunct_island", "anaphor_agreement", "argument_structure",
            "binding", "control_raising", "determiner_noun_agreement",
            "ellipsis", "filler_gap", "irregular_forms", "npi_licensing",
            "quantifiers", "subject_verb_agreement", "tough_movement",
            "wh_questions", "wh_vs_that_with_gap", "wh_vs_that_no_gap",
            "wh_vs_that_with_gap_long_distance", "wh_vs_that_no_gap_long_distance",
        ]

def run_subset(model, tok, subset: str, split: str, device: torch.device, limit: int, normalize: str, dump: int = 0):
    """Run PLL evaluation on one BLiMP subset split.

    Returns a dict with fields: subset, split, n, acc, oov_rate, normalize, examples.
    """
    ds = load_dataset("blimp", subset, split=split)
    n = min(limit, len(ds)) if limit > 0 else len(ds)
    wins = 0
    total = 0
    oov_tokens = 0
    total_tokens = 0
    gaps: List[Tuple[float, Dict[str, Any]]] = []  # (good-bad gap, info)

    for row in tqdm(ds.select(range(n)), desc=f"{subset}/{split}"):
        good = row["sentence_good"]
        bad = row["sentence_bad"]
        s_good, contrib_g, enc_g = ar_score_with_tokens(model, tok, good, device, normalize=normalize)
        s_bad,  contrib_b, enc_b = ar_score_with_tokens(model, tok, bad,  device, normalize=normalize)
        gap = s_good - s_bad
        wins += int(gap > 0.0)
        total += 1
        # OOV stats
        unk = tok.unk_token_id
        if unk is not None and unk >= 0:
            oov_tokens += int((enc_g["input_ids"] == unk).sum().item() + (enc_b["input_ids"] == unk).sum().item())
        total_tokens += int(enc_g["input_ids"].numel() + enc_b["input_ids"].numel())
        if dump > 0:
            gaps.append((gap, {
                "good": good,
                "bad": bad,
                "gap": gap,
                "s_good": s_good,
                "s_bad": s_bad,
                "good_top_neg_tokens": sorted(contrib_g, key=lambda t: t[2])[:3],
                "bad_top_pos_tokens": sorted(contrib_b, key=lambda t: -t[2])[:3],
            }))

    acc = wins / max(1, total)
    oov_rate = oov_tokens / max(1, total_tokens)
    worst: List[Dict[str, Any]] = []
    if dump > 0 and gaps:
        worst = [info for _, info in sorted(gaps, key=lambda x: x[0])[:dump]]
    return {
        "subset": subset,
        "split": split,
        "n": total,
        "acc": acc,
        "oov_rate": oov_rate,
        "normalize": normalize,
        "examples": worst,
    }

def main():
    """CLI entry point to evaluate a model on BLiMP and/or BLiMP-supplement.

    Example
        python blimp_sanity.py --model_path ./model_vault/model_bl_bert_ltgds_regular \
            --benchmark both --max_examples 200 --normalize per_token --dump 5
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--subset", default=None, help="If unset, run all BLiMP subsets available")
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="auto", help="auto | train | test | validation")
    ap.add_argument("--normalize", default="none", choices=["none", "per_token"])
    ap.add_argument("--dump", type=int, default=5, help="dump worst-K examples per subset")
    ap.add_argument("--save_csv", default="outputs/blimp_pll_report.csv")
    ap.add_argument("--save_json", default="outputs/blimp_pll_examples.jsonl")
    # New: supplement options
    ap.add_argument("--benchmark", default="blimp", choices=["blimp", "blimp_supplement", "both"],
                    help="Which benchmark(s) to run")
    ap.add_argument("--supplement_dir", default=None, help="Path to folder containing BLiMP supplement JSONL files")
    ap.add_argument("--supplement_task", default=None, help="If set, only run this supplement task (filename without .jsonl)")
    ap.add_argument("--supplement_save_csv", default="outputs/blimp_supplement_pll_report.csv")
    ap.add_argument("--supplement_save_json", default="outputs/blimp_supplement_pll_examples.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.supplement_save_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.supplement_save_json), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Tokenizer: ensure pad_token set for GPT-2-like models
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Fast tokenizer load failed: {e}\n[INFO] Falling back to use_fast=False")
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    # Load causal LM (e.g., GPT-2)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device).eval()

    # Run BLiMP (HF) benchmark if requested
    rows: List[Dict[str, Any]] = []
    if args.benchmark in ("blimp", "both"):
        subsets = [args.subset] if args.subset else ensure_subsets_list("blimp")
        with open(args.save_json, "w") as jf:
            for subset in subsets:
                try:
                    split = pick_split("blimp", subset, args.split)
                except Exception as e:
                    print(f"[WARN] Skipping {subset}: {e}")
                    continue
                res = run_subset(model, tok, subset, split, device, args.max_examples, args.normalize, dump=args.dump)
                # Print accuracy as soon as the subset is processed
                print(f"Subset={subset:<35} split={split:<10} n={res['n']:<5} PLL-acc={res['acc']:.4f}  OOV={res['oov_rate']:.2%}", flush=True)
                rows.append(res)
                for ex in res.get("examples", []):
                    jf.write(json.dumps({"subset": subset, "split": split, **ex}, ensure_ascii=False) + "\n")
        with open("outputs/blimp_pll_summary.jsonl", "w") as sf:
            for r in rows:
                sf.write(json.dumps({
                    "subset": r["subset"],
                    "split": r["split"],
                    "acc": r["acc"],
                    "oov_rate": r["oov_rate"],
                    "n": r["n"]
                }) + "\n")
                print(f"Subset={r['subset']:<35} split={r['split']:<10} n={r['n']:<5} PLL-acc={r['acc']:.4f}  OOV={r['oov_rate']:.2%}")
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subset", "split", "n", "acc", "oov_rate", "normalize"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        if rows:
            macro = sum(r["acc"] for r in rows) / len(rows)
            print(f"\n[BLiMP Summary] subsets={len(rows)}  macro-PLL-acc={macro:.4f}  saved: {args.save_csv}, {args.save_json}")

    # Run BLiMP supplement benchmark if requested
    rows_sup: List[Dict[str, Any]] = []
    if args.benchmark in ("blimp_supplement", "both"):
        supp_dir = find_blimp_supplement_dir(args.supplement_dir)
        if not supp_dir:
            raise RuntimeError("Could not locate BLiMP supplement folder. Pass --supplement_dir pointing to JSONL files.")
        tasks = [args.supplement_task] if args.supplement_task else get_available_supplement_tasks(supp_dir)
        if not tasks:
            raise RuntimeError(f"No supplement tasks found in {supp_dir}")
        with open(args.supplement_save_json, "w") as jf:
            for task in tasks:
                fpath = os.path.join(supp_dir, f"{task}.jsonl")
                if not os.path.isfile(fpath):
                    print(f"[WARN] Supplement file missing, skipping: {fpath}")
                    continue
                res = run_supplement_subset(model, tok, task, fpath, device, args.max_examples, args.normalize, dump=args.dump)
                # Print accuracy as soon as the supplement task is processed
                print(f"SuppTask={task:<28} n={res['n']:<5} PLL-acc={res['acc']:.4f}  OOV={res['oov_rate']:.2%}", flush=True)
                rows_sup.append(res)
                for ex in res.get("examples", []):
                    jf.write(json.dumps({"task": task, **ex}, ensure_ascii=False) + "\n")
        # summary + csv
        with open("outputs/blimp_supplement_pll_summary.jsonl", "w") as sf:
            for r in rows_sup:
                sf.write(json.dumps({
                    "task": r["task"],
                    "acc": r["acc"],
                    "oov_rate": r["oov_rate"],
                    "n": r["n"],
                    "source": r.get("source", "")
                }) + "\n")
                print(f"SuppTask={r['task']:<28} n={r['n']:<5} PLL-acc={r['acc']:.4f}  OOV={r['oov_rate']:.2%}")
        with open(args.supplement_save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "n", "acc", "oov_rate", "normalize", "source"])
            w.writeheader()
            for r in rows_sup:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        if rows_sup:
            macro_sup = sum(r["acc"] for r in rows_sup) / len(rows_sup)
            print(f"\n[BLiMP Supplement Summary] tasks={len(rows_sup)}  macro-PLL-acc={macro_sup:.4f}  saved: {args.supplement_save_csv}, {args.supplement_save_json}")

    # quick overall print if both
    if args.benchmark == "both":
        total_tasks = (len(rows) if rows else 0) + (len(rows_sup) if rows_sup else 0)
        if total_tasks:
            macro_all = (sum(r["acc"] for r in rows) + sum(r["acc"] for r in rows_sup)) / total_tasks
            print(f"\n[Overall] total_parts={total_tasks} macro-PLL-acc={macro_all:.4f}")

if __name__ == "__main__":
    main()