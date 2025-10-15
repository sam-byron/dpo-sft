from datetime import datetime
from collections import OrderedDict
import gc
import math
# from torch.utils.data import Dataset
import os
from pathlib import Path
import random
import glob
from functools import partial
import hashlib
import pickle
from multiprocessing import Pool
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
import time
import argparse
import json
from typing import Union
from typing import List, Iterator, Any
import sys
from transformers import DataCollatorForLanguageModeling

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

STRUCTURAL_TOKENS = ["[PAD]","[UNK]","[MASK]","[PAR]","[TAB]","[SPK]","<|endoftext|>"]
def get_special_token_ids(tokenizer):
    ids = []
    for tok in STRUCTURAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None:
            ids.append(tid)
    return sorted(set(ids))

# helper to compute total real tokens in a Dataset
def compute_total_tokens(ds: Union[Dataset, object]) -> int:
    """
    If `ds` has an index_map of (path, seq_idx, start, end) tuples,
    sum up (end - start). Otherwise assume __getitem__ returns a dict
    with 'input_ids' and sum their lengths.
    """
    if hasattr(ds, "index_map"):
        return sum(end - start for (_, _, start, end) in ds.index_map)
    # fallback for map‐style datasets returning dicts
    return sum(len(ex) for ex in ds)

class ChunkedDataset(Dataset):
    """Dataset over pre-generated fixed-size training blocks stored in chunk*.pt files.

    Two operational modes:
      1. Preblocked mode (preferred): Each chunk file is a 2-D LongTensor [N, block_size]. We simply
         index rows without any re-concatenation.
      2. Legacy concatenation mode: Each chunk file stores a list/1-D sequences requiring chaining
         into uniform blocks (old pipeline). Retained for backward compatibility.
    """
    def __init__(self, chunk_paths, block_size, tokenizer, dtype=torch.long, pad_token_id=None, cache_size=50):
        self.chunk_paths = list(chunk_paths)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.pad_token_id = pad_token_id or 0
        self.cache_size = cache_size
    # path -> loaded tensor (2-D) OR concatenated flat list (legacy)
        self._chunk_cache: "OrderedDict[str, Any]" = OrderedDict()

        # Defer cache_dir until we know dataset root; place under chunk folder for stability

        # Detect format from first available file
        sample_path = None
        for pth in self.chunk_paths:
            if os.path.exists(pth):
                sample_path = pth
                break
        if sample_path is None:
            raise ValueError("No existing chunk paths provided to ChunkedDataset")

        # Set cache directory relative to dataset root for reproducibility
        root_dir = Path(sample_path).parent
        self.cache_dir = root_dir / "cache_index"
        self.cache_dir.mkdir(exist_ok=True)

        sample_obj = torch.load(sample_path, map_location="cpu")
        self.preblocked = torch.is_tensor(sample_obj) and sample_obj.dim() == 2
        if self.preblocked and self.block_size is not None and sample_obj.shape[1] != self.block_size:
            raise ValueError(f"Block size mismatch: arg {self.block_size} vs file width {sample_obj.shape[1]}")
        if self.preblocked:
            # Standard fast path
            self._build_index_preblocked()

    def _get_cache_key(self):
        """Generate a cache key based on chunk paths, modification times, and block size."""
        # Create a hash of all chunk paths and their modification times
        path_data = []
        for path in sorted(self.chunk_paths):
            try:
                mtime = os.path.getmtime(path)
                path_data.append((path, mtime))
            except OSError:
                # If file doesn't exist, use current timestamp
                path_data.append((path, time.time()))
        
        # Include block size in the hash
        cache_input = (tuple(path_data), self.block_size)
        cache_str = str(cache_input).encode('utf-8')
        return hashlib.md5(cache_str).hexdigest()
    


    def _build_index_preblocked(self):
        """Index rows directly for preblocked 2-D tensors."""
        cache_key = self._get_cache_key() + "_preblocked"
        cache_file = self.cache_dir / f"index_map_{cache_key}.pkl"
        if cache_file.exists():
            try:
                print(f"{C.CYAN}Loading cached (preblocked) index map from {cache_file}{C.RESET}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.index_map = cached_data['index_map']
                self.block_lengths = cached_data['block_lengths']
                print(f"{C.GREEN}Loaded {len(self.index_map)} row entries from cache{C.RESET}")
                return
            except Exception as e:
                print(f"{C.YELLOW}Cache load failed, rebuilding preblocked index: {e}{C.RESET}")
                try: cache_file.unlink()
                except Exception: pass
        print(f"{C.BLUE}Building preblocked index (row-level) ...{C.RESET}")
        self.index_map = []
        self.block_lengths = []
        total_rows = 0
        for pth in self.chunk_paths:
            try:
                tensor = torch.load(pth, map_location='cpu')
                if not (torch.is_tensor(tensor) and tensor.dim() == 2):
                    raise ValueError("Encountered non-2D tensor in preblocked mode; mixed formats not supported")
                n_rows, width = tensor.shape
                if width != self.block_size:
                    raise ValueError(f"Width mismatch in {pth}: expected {self.block_size} got {width}")
                # store (path, row_idx, start, end) with synthetic offsets for compatibility
                for r in range(n_rows):
                    start = r * self.block_size
                    end = start + self.block_size
                    self.index_map.append((pth, r, start, end))
                    self.block_lengths.append(self.block_size)
                total_rows += n_rows
            except Exception as e:
                print(f"{C.RED}Failed indexing {pth}: {e}{C.RESET}")
        print(f"{C.MAGENTA}Preblocked stats:{C.RESET} rows={total_rows:,} block_size={self.block_size}")
        # Cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'index_map': self.index_map, 'block_lengths': self.block_lengths}, f)
            print(f"{C.GREEN}Cached preblocked index to {cache_file}{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}Warning: failed to cache preblocked index: {e}{C.RESET}")

    
    def _cleanup_old_cache_files(self, keep_latest=5):
        """Remove old cache files, keeping only the most recent ones."""
        try:
            cache_files = list(self.cache_dir.glob("index_map_*.pkl"))
            if len(cache_files) <= keep_latest:
                return
            
            # Sort by modification time, newest first
            cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove old files
            for old_file in cache_files[keep_latest:]:
                try:
                    old_file.unlink()
                    print(f"{C.DIM}Removed old cache file: {old_file.name}{C.RESET}")
                except OSError:
                    pass
        except Exception as e:
            print(f"{C.YELLOW}Warning: Failed to cleanup cache files: {e}{C.RESET}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        path, row_or_seq, start, end = self.index_map[idx]
        if self.preblocked:
            # Load entire 2-D tensor into cache lazily
            if path not in self._chunk_cache:
                tensor = torch.load(path, map_location='cpu')
                if not (torch.is_tensor(tensor) and tensor.dim() == 2):
                    raise ValueError(f"Expected 2-D tensor in preblocked mode for {path}")
                self._chunk_cache[path] = tensor
                if len(self._chunk_cache) > self.cache_size:
                    self._chunk_cache.popitem(last=False)
            tensor = self._chunk_cache[path]
            self._chunk_cache.move_to_end(path, last=True)
            block = tensor[row_or_seq].clone()
            # Ensure dtype long
            if block.dtype != torch.long:
                block = block.long()
            return block

# Update the data_loader function to use sentence-aware dataset when available
def data_loader(config, tokenizer, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    
    # Check if we should use sentence-aware processing
    use_sentence_aware = config.get("use_sentence_aware", True)
    
    if use_sentence_aware:
        print(f"{C.MAGENTA}🎯 Using Sentence-Aware Dataset for improved syntax learning{C.RESET}")
        return _create_sentence_aware_loader(config, tokenizer, cache_path)

def _create_sentence_aware_loader(config, tokenizer, cache_path):
    """Create data loaders using the sentence-aware dataset."""
    from dataset import SentenceAwareDataset
    
    # Create the sentence-aware dataset
    print(f"{C.CYAN}Loading sentence-aware dataset from {cache_path}...{C.RESET}")
    
    try:
        # Use the new sentence-aware dataset
        full_dataset = SentenceAwareDataset(
            cache_path=cache_path,
            tokenizer=tokenizer,
            seq_length=config.get("block_size", 512),
        )

        print(f"{C.GREEN}Loaded {len(full_dataset)} sentences for sentence-aware training{C.RESET}")

        # Split the dataset (respect config fractions if provided)
        total_size = len(full_dataset)
        train_frac = float(config.get("train_frac", 0.5))
        val_frac = float(config.get("val_frac", 0.01))
        # Ensure sane bounds
        train_frac = min(max(train_frac, 0.0), 0.99)
        val_frac = min(max(val_frac, 0.0), 1.0 - train_frac)
        train_size = int(train_frac * total_size)
        val_size = int(val_frac * total_size)
        test_size = total_size - train_size - val_size

        # Create train/val/test splits
        import torch.utils.data as data
        # Optional: reproducible split
        gen = None
        if "data_split_seed" in config and isinstance(config["data_split_seed"], int):
            gen = torch.Generator()
            gen.manual_seed(int(config["data_split_seed"]))
        train_dataset, val_dataset, test_dataset = data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=gen
        )

        # Optionally merge test into validation to increase validation stability
        merge_test_into_val = bool(config.get("merge_test_into_val", False))
        if merge_test_into_val and len(test_dataset) > 0:
            # Construct a new Subset combining val and test indices
            try:
                combined_indices = list(val_dataset.indices) + list(test_dataset.indices)
                val_dataset = data.Subset(full_dataset, combined_indices)
                test_dataset = data.Subset(full_dataset, [])  # empty test
                print(f"{C.YELLOW}merge_test_into_val=True → New val size: {len(val_dataset)}, test size: {len(test_dataset)}{C.RESET}")
            except Exception as e:
                print(f"{C.YELLOW}merge_test_into_val requested but failed to merge: {e}. Proceeding without merge.{C.RESET}")

        print(f"{C.CYAN}Dataset splits → train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}{C.RESET}")

        collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,  # moderate multiprocessing
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )

        # Compute total tokens precisely (train split)
        if hasattr(full_dataset, "sequences") and hasattr(train_dataset, "indices"):
            # Fast path: sum lengths directly from underlying storage using subset indices
            total_tokens_train = sum(int(len(full_dataset.sequences[i])) for i in train_dataset.indices)
            avg_sentence_length = total_tokens_train / max(1, len(train_dataset))
        else:
            # Fallback: iterate the subset (slower but exact)
            total_tokens_train = 0
            for item in train_dataset:
                seq = item['input_ids'] if isinstance(item, dict) else item
                total_tokens_train += int(len(seq))
            avg_sentence_length = total_tokens_train / max(1, len(train_dataset))

        print(f"{C.GREEN}Sentence-aware data loaders created successfully{C.RESET}")
        print(f"{C.CYAN}Average sentence length: {avg_sentence_length:.1f} tokens{C.RESET}")
        print(f"{C.CYAN}Total training tokens (exact): {total_tokens_train:,}{C.RESET}")

        return train_loader, val_loader, test_loader, collator, total_tokens_train

    except Exception as e:
        print(f"{C.RED}Error creating sentence-aware dataset: {e}{C.RESET}")
