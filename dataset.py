import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import gzip
from typing import Optional, Sequence


def _load_chunk_sentences(chunk_path):
    """Helper function for multiprocessing chunk loading - robust version with memory management."""
    try:
        import torch
        import os
        import gc
        
        # Get file info
        filename = os.path.basename(chunk_path)
        file_size = os.path.getsize(chunk_path)
        
        # Load chunk data with memory optimization
        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        sentences = []
        
        # Process sentences efficiently
        if isinstance(chunk_data, torch.Tensor):
            # Preblocked format: [num_rows, block_size] or [block_size]
            if chunk_data.ndim == 2:
                for i in range(chunk_data.size(0)):
                    row = chunk_data[i]
                    if row.numel() > 0:
                        if row.dtype != torch.long:
                            row = row.long()
                        sentences.append(row)
            elif chunk_data.ndim == 1 and chunk_data.numel() > 0:
                row = chunk_data
                if row.dtype != torch.long:
                    row = row.long()
                sentences.append(row)

        
        # Aggressive cleanup to prevent memory leaks
        del chunk_data
        gc.collect()
        
        # Progress indicator
        print(f"✓ {filename}: {len(sentences)} sentences ({file_size/(1024*1024):.1f}MB)")
        
        return sentences
        
    except Exception as e:
        print(f"❌ Error loading {chunk_path}: {e}")
        # Force cleanup on error
        import gc
        gc.collect()
        return []  # Return empty list instead of None for easier handling


class Indexer:
    def __init__(self, documents):
        lengths = [len(document) for document in documents]
        self.cumsum = torch.LongTensor([0] + lengths).cumsum(dim=0)

    def get_indices(self, index):
        document_index = torch.searchsorted(self.cumsum, index, right=True).item() - 1
        segment_index = index - self.cumsum[document_index]
        return document_index, segment_index

    def __len__(self):
        return self.cumsum[-1].item()


class SentenceAwareDataset(Dataset):
    """Dataset that loads tokenized sentences with preserved boundaries from PT files."""
    
    def __init__(self, cache_path, tokenizer, seq_length=512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.pad_token_id

        # Dynamically determine structural special token span (must be contiguous prefix 0..N-1)
        # structural_tokens = ["[PAD]","[UNK]","[SPK]"]
        structural_tokens = ["[SPK]"]
        special_ids = []
        for tok in structural_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None:
                special_ids.append(tid)
        special_ids = sorted(set(special_ids))
        if special_ids and special_ids == list(range(len(special_ids))):
            self.n_special_tokens = len(special_ids)
        else:
            self.n_special_tokens = 6  # legacy fallback
            if special_ids:
                print(f"[SentenceAwareDataset] Warning: non-contiguous structural token IDs {special_ids}; fallback n_special_tokens=6")
        self.padding_label_id = -100

        
        # Simple on-disk cache for prebuilt sequences
        import os
        self._rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
        self._is_main = (self._rank == 0)
        cache_dir = os.path.join(cache_path, "cache_index")
        os.makedirs(cache_dir, exist_ok=True)
        self._seq_cache_path = os.path.join(cache_dir, f"sentence_sequences_seq{self.seq_length}.pt")

        # Fast path: load cached sequences if present
        if os.path.exists(self._seq_cache_path):
            if self._is_main:
                print(f"[SentenceAwareDataset] Using cached sequences: {self._seq_cache_path}")
            import torch
            self.sequences = torch.load(self._seq_cache_path, map_location="cpu")
            # Done: skip chunk scanning/sequence build
            return
        # Load all chunk files with robust multiprocessing
        import glob
        import os
        from multiprocessing import Pool, cpu_count
        import multiprocessing as mp
        from functools import partial
        
        chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
        print(f"Loading {len(chunk_paths)} chunk files for sentence-aware MLM training...")
        
        if len(chunk_paths) == 0:
            raise ValueError(f"No chunk files found in {cache_path}")
        
        self.sentences = []
        
        # Use simple sequential loading for reliability
        print(f"💾 Memory before loading: {self._get_memory_usage()}")
        
        for i, path in enumerate(chunk_paths):
            try:
                result = _load_chunk_sentences(path)
                if result:
                    self.sentences.extend(result)
                if (i + 1) % 10 == 0:
                    print(f"Sequential: {i + 1}/{len(chunk_paths)} chunks loaded, {len(self.sentences)} sentences so far")
                    print(f"💾 Memory usage: {self._get_memory_usage()}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"✅ Loaded {len(self.sentences)} sentences from {len(chunk_paths)} chunks")
        
        # After building sequences (success path)
        try:
            self.sequences = self._create_multi_sentence_sequences()
            print(f"✅ Successfully created {len(self.sequences)} sequences")
        except Exception as e:
            print(f"❌ Error creating sequences: {e}")
            # Fallback: create simple sequences from individual sentences
            print("Creating fallback individual sentence sequences...")
            self.sequences = self.sentences[:10000]  # Use first 10k sentences as sequences
            print(f"✅ Created {len(self.sequences)} fallback sequences")

        # Save sequences to cache (rank 0 only)
        try:
            if self._is_main:
                import torch, tempfile, shutil
                tmp_path = self._seq_cache_path + ".tmp"
                torch.save(self.sequences, tmp_path)
                os.replace(tmp_path, self._seq_cache_path)  # atomic on POSIX
                print(f"[SentenceAwareDataset] Cached sequences to {self._seq_cache_path}")
        except Exception as e:
            if self._is_main:
                print(f"[SentenceAwareDataset] Warning: failed to cache sequences: {e}")

        # EMERGENCY: Clean up sentences list to save memory AFTER sequence creation
        print("🧹 Cleaning up intermediate sentences data to save memory...")
        temp_sequences_count = len(self.sequences)
        del self.sentences  # Free memory from sentences list
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        print(f"✅ Memory cleanup complete. Dataset ready with {temp_sequences_count} sequences")
        
    def _get_memory_usage(self):
        """Get current memory usage in a readable format."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        except:
            return "unknown"
    
    def _create_multi_sentence_sequences(self):
        """Combine multiple sentences into sequences up to seq_length tokens."""
        print("Creating multi-sentence sequences for long-range learning...")
        sequences = []
        current_sequence = []
        current_length = 0
        
        # Pre-allocate pad tensor for efficiency
        pad_tensor_cache = {}
        
        for i, sentence in enumerate(self.sentences):
            sentence_length = len(sentence)
            
            # Truncate long sentences instead of skipping them - preserve all data!
            if sentence_length > self.seq_length:
                if i % 10000 == 0:  # Only log every 10,000th truncation to reduce spam
                    print(f"Truncating sentence {i} from length {sentence_length} to {self.seq_length}")
                sentence = sentence[:self.seq_length]  # Truncate to max length
                sentence_length = self.seq_length
            
            # If adding this sentence would exceed seq_length, save current and start new
            if current_length + sentence_length > self.seq_length and current_sequence:
                # Combine current sequence
                combined = torch.cat(current_sequence)
                padding_needed = self.seq_length - len(combined)
                
                if padding_needed > 0:
                    # Use cached padding tensor for efficiency
                    if padding_needed not in pad_tensor_cache:
                        pad_tensor_cache[padding_needed] = torch.full((padding_needed,), self.pad_token_id, dtype=torch.long)
                    combined = torch.cat([combined, pad_tensor_cache[padding_needed]])
                
                sequences.append(combined)
                current_sequence = []
                current_length = 0
                
                # Progress update
                if len(sequences) % 10000 == 0:
                    print(f"Created {len(sequences)} sequences so far...")
            
            # Add sentence to current sequence
            current_sequence.append(sentence)
            current_length += sentence_length
        
        # Add final sequence if it exists
        if current_sequence:
            combined = torch.cat(current_sequence)
            padding_needed = self.seq_length - len(combined)
            if padding_needed > 0:
                if padding_needed not in pad_tensor_cache:
                    pad_tensor_cache[padding_needed] = torch.full((padding_needed,), self.pad_token_id, dtype=torch.long)
                combined = torch.cat([combined, pad_tensor_cache[padding_needed]])
            sequences.append(combined)
        
        print(f"✅ Created {len(sequences)} multi-sentence sequences")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a multi-sentence training example with MLM masking."""
        sequence = self.sequences[idx].clone()
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (sequence != self.pad_token_id).bool()
        labels = torch.full_like(sequence, self.padding_label_id)
        
        return {
            'input_ids': sequence,
            'attention_mask': attention_mask,
            'labels': labels
        }

if __name__ == "__main__":
    # Minimal smoke test for SentenceAwareDataset only if desired; guarded to avoid heavy load.
    pass
