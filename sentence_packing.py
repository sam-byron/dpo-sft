"""Sentence packing utilities.

Packs individual sentences into sequences up to a token length budget
so that fewer segment boundaries fall mid-sentence. Uses a lightweight
regex-based sentence splitter to avoid heavy dependencies.

Output format (pure GPT-2 style - no explicit separators):
  Sentence one. Sentence two? Sentence three!

If a single sentence exceeds the max length budget (after accounting for
special tokens) it is chunked into sliding windows (hard split) to avoid
dropping content.
"""
from __future__ import annotations
import re
from typing import List, Iterable, Tuple, Dict, Any

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SPEAKER_RE = re.compile(r"^([A-Z][a-zA-Z]{1,20}):")

FILLER_TERMINALS = {"yeah","mm","erm","uh","er","ah","hmm"}

def simple_sentence_tokenize(text: str) -> List[str]:
    """Very lightweight sentence splitter.
    Splits on punctuation + whitespace boundaries. If no split found, returns [text].
    Trims whitespace. Keeps ending punctuation attached.
    """
    text = text.strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def pack_sentences(lines: Iterable[str], max_length: int, min_fill_ratio: float = 0.85,
                   ensure_terminal_punct: bool = True, max_dangling_tokens: int = 48,
                   allow_borrow: bool = True, borrow_first_sent_max_tokens: int = 8,
                   add_doc_and_speaker: bool = True,
                   collect_stats: bool = False,
                   drop_short_trailing: bool = False,
                   short_trailing_max_tokens: int = 10,
                   first_names: set = None) -> List[str]:
    """Pack sentences into sequences (pure GPT-2 style - no separators).

    Args:
        lines: iterable of raw text lines (may contain multiple sentences)
        tokenizer: tokenizer with encode() returning .ids or a list directly
        max_length: maximum total token length (including specials)
        min_fill_ratio: when finishing a pack, if used length < ratio*max_length,
                        attempt to pull one more sentence if it fits poorly truncated.
        first_names: set of first names for enhanced speaker detection
    Returns:
        List of packed string samples.
    """
    spk_token = "[SPK]"
    # Use GPT-2's native <|endoftext|> token for document END boundaries only (not start)
    eod_token = "<|endoftext|>"

    # Helper function to get token IDs from encode result (handles both HF and tokenizers library)
    def get_ids(text: str) -> List[str]:
        # Disable length warning since we handle chunking ourselves
        result = text.split()
        if isinstance(result, list):
            return result
        elif hasattr(result, 'ids'):
            return result.ids
        else:
            raise TypeError(f"Unexpected tokenizer.encode() return type: {type(result)}")
    
    # Helper to get token ID for a special token
    def get_token_id(token: str) -> str:
        return token

    # Enhanced speaker detection using both pattern matching and first names
    def detect_speaker(line: str) -> bool:
        """Detect if a line starts with a speaker identifier."""
        if not line:
            return False
        
        # Pattern 1: "Name: text" or "Name — text" or "Name - text"
        match = SPEAKER_RE.match(line)
        if match:
            speaker_name = match.group(1)
            # If we have first names list, verify it's a known name
            if first_names:
                return speaker_name in first_names
            return True
        
        # Pattern 2: Check if line starts with a capitalized word followed by colon/dash
        # and that word is in our first names list
        if first_names:
            words = line.split(None, 2)  # Split on whitespace, max 2 splits
            if len(words) >= 2:
                first_word = words[0].rstrip(':—-')
                if first_word in first_names and any(c in words[0] for c in ':—-'):
                    return True
        
        return False

    have_spk = get_token_id(spk_token) is not None
    have_eod = get_token_id(eod_token) is not None
    
    if not have_spk:
        print(f"[SpeakerTag] Warning: {spk_token} token not found in tokenizer vocabulary")
    if not have_eod:
        print(f"[DocBoundary] Warning: {eod_token} token not found in tokenizer vocabulary")
    
    if first_names:
        print(f"[SpeakerTag] Enhanced speaker detection enabled with {len(first_names)} first names")

    stats = {} if collect_stats else None
    if stats is not None:
        stats['packs'] = 0
        stats['terminal_end'] = 0
        stats['nonterminal_end'] = 0
        stats['borrow_moves'] = 0
        stats['dangling_carried'] = 0
        stats['short_trailing_dropped'] = 0
        stats['speaker_tags_added'] = 0  # New stat

    packed: List[str] = []
    current_sentences: List[str] = []
    current_len: int = 0

    def is_sentence_terminal(sentence: str) -> bool:
        if not sentence:
            return False
        if sentence[-1] in '.!?':
            return True
        low = sentence.lower().strip("'\"")
        if low in FILLER_TERMINALS:
            return True
        return False

    def flush(force: bool=False):
        nonlocal current_sentences, current_len
        if not current_sentences:
            return
        # If we want punctuation-final endings, attempt to carry over a dangling sentence.
        if ensure_terminal_punct and not force:
            # If last sentence lacks terminal punctuation and is short, carry it to next pack.
            last = current_sentences[-1]
            if (last and not is_sentence_terminal(last)):
                # Estimate token count (rough); if short, pop and delay.
                last_ids = get_ids(last)
                if len(last_ids) <= max_dangling_tokens:
                    if stats is not None:
                        stats["dangling_carried"] += 1
                    dangling = current_sentences.pop()
                    # Only flush remaining if any remain.
                    if current_sentences:
                        sample = " ".join(current_sentences)
                        packed.append(sample)
                        if stats is not None:
                            last_sent = current_sentences[-1]
                            if is_sentence_terminal(last_sent):
                                stats["terminal_end"] += 1
                            else:
                                stats["nonterminal_end"] += 1
                            stats["packs"] += 1
                    # Start next pack with dangling sentence.
                    current_sentences = [dangling]
                    current_len = len(get_ids(dangling))
                    return
        # Normal flush
        if drop_short_trailing and not force and current_sentences:
            last = current_sentences[-1]
            if not is_sentence_terminal(last):
                last_ids = get_ids(last)
                fill_ratio = current_len / max_length
                if len(last_ids) <= short_trailing_max_tokens and fill_ratio >= min_fill_ratio:
                    # Drop last and carry forward
                    current_sentences.pop()
                    if stats is not None:
                        stats["short_trailing_dropped"] += 1
                    if current_sentences:
                        sample = " ".join(current_sentences)
                        packed.append(sample)
                        if stats is not None:
                            last_sent = current_sentences[-1]
                            if is_sentence_terminal(last_sent):
                                stats["terminal_end"] += 1
                            else:
                                stats["nonterminal_end"] += 1
                            stats["packs"] += 1
                    # Start new pack buffer with dropped fragment
                    current_sentences = [last]
                    current_len = len(last_ids)
                    return
        sample = " ".join(current_sentences)
        packed.append(sample)
        if stats is not None:
            last_sent = current_sentences[-1]
            if is_sentence_terminal(last_sent):
                stats["terminal_end"] += 1
            else:
                stats["nonterminal_end"] += 1
            stats["packs"] += 1
        current_sentences = []
        current_len = 0

    # Preprocess to inject doc boundaries and speaker markers if requested
    processed_lines: List[str] = []
    if add_doc_and_speaker:
        # Detect document boundaries: lines list may contain explicit markers <|endoftext|>
        # GPT-2 native way: <|endoftext|> appears only at document END
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # <|endoftext|> marks end of document (appears as a standalone line or at end of text)
            if line == eod_token:
                processed_lines.append(line if have_eod else line)
                continue
            # Enhanced speaker tagging using first names
            if detect_speaker(line) and have_spk:
                line = f"{spk_token} {line}"
                if stats is not None:
                    stats['speaker_tags_added'] += 1
            processed_lines.append(line)
    else:
        processed_lines = [l for l in lines if l.strip()]

    for line in processed_lines:
        # Split line into sentences
        sentences = simple_sentence_tokenize(line)
        if not sentences:
            continue
        for sent in sentences:
            # token length for this sentence (no separator needed)
            sent_ids = get_ids(sent)
            sent_len = len(sent_ids)
            # If this single sentence is too long to fit, chunk it
            budget = max_length
            if sent_len > max_length:  # > max_length
                # Flush what we have
                flush()
                # Hard chunk sentence tokens
                tokens = sent_ids
                window = budget
                start = 0
                while start < len(tokens):
                    piece_ids = tokens[start:start+window]
                    piece_text = "".join(piece_ids).strip()
                    packed.append(piece_text)
                    start += window
                continue

            # Normal case: attempt to add sentence to current pack
            prospective = current_len + sent_len
            if prospective > max_length:
                # Decide whether to flush or try to fill more (if current pack badly under-filled) — we flush now
                flush()
                current_sentences.append(sent)
                current_len = sent_len  # just this sentence
            else:
                current_sentences.append(sent)
                current_len = prospective

        # After processing a line, if we are far from filling, continue; else consider flush opportunistically
        if current_len / max_length >= min_fill_ratio:
            flush()

    # Flush remainder
    flush(force=True)

    # Backward borrow pass: attempt to move a short first sentence from a pack to previous
    if allow_borrow and len(packed) > 1:
        adjusted: List[str] = []
        prev_sent_lists: List[List[str]] = []
        # Recover sentence lists from packed strings
        def unpack(sample: str) -> List[str]:
            # sample pattern: S1. S2? S3!
            # Use the same sentence splitter as input
            return simple_sentence_tokenize(sample)
        sent_lists = [unpack(p) for p in packed]
        changed = False
        for i in range(1, len(sent_lists)):
            prev_list = sent_lists[i-1]
            curr_list = sent_lists[i]
            if not prev_list or not curr_list:
                continue
            last_prev = prev_list[-1]
            first_curr = curr_list[0]
            if not is_sentence_terminal(last_prev) and is_sentence_terminal(first_curr):
                # token count check
                prev_ids = []
                for s in prev_list:
                    prev_ids.extend(get_ids(s))
                first_ids = get_ids(first_curr)
                projected = len(prev_ids) + len(first_ids)
                if projected <= max_length and len(first_ids) <= borrow_first_sent_max_tokens:
                    # move
                    prev_list.append(first_curr)
                    del curr_list[0]
                    changed = True
                    if stats is not None:
                        stats["borrow_moves"] += 1
        if changed:
            # Rebuild packed strings
            rebuilt = []
            for sl in sent_lists:
                if not sl:
                    continue
                rebuilt.append(" ".join(sl))
            packed = rebuilt

    # Final strict length enforcement
    final_packed = []
    for sample in packed:
        ids = get_ids(sample)
        if len(ids) <= max_length:
            final_packed.append(sample)
        else:
            # Attempt to drop last sentence and retry
            sents = simple_sentence_tokenize(sample)
            if len(sents) > 1:
                trimmed = sents[:-1]
                new_sample = " ".join(trimmed)
                ids2 = get_ids(new_sample)
                if len(ids2) <= max_length:
                    final_packed.append(new_sample)
                else:
                    # Drop entirely if still too long (rare)
                    continue
            else:
                continue
    if stats is not None:
        return final_packed, stats
    return final_packed

__all__ = ["pack_sentences", "simple_sentence_tokenize"]
