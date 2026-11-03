"""Byte Pair Encoding (BPE): how modern language models actually tokenize.

Word vocabularies cannot represent unseen words; character vocabularies make
sequences long. BPE learns the middle ground from data:

    1. Start from raw bytes, ids 0-255. Any text is representable, forever.
    2. Find the most frequent adjacent pair of symbols.
    3. Merge that pair into a brand-new symbol.
    4. Repeat until the vocabulary reaches the requested size.

Merges are ordered, and that order is the model. Encoding replays them,
always applying the earliest-learned merge available.
"""
from __future__ import annotations

import re
from collections import Counter

# Pre-split so merges never cross a word boundary. The leading space stays
# attached to the word: " the" and "the" are different tokens, which is how
# detokenizing can be exact.
_CHUNK_RE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


def _pair_counts(chunks: Counter) -> Counter:
    counts: Counter = Counter()
    for chunk, freq in chunks.items():
        for pair in zip(chunk, chunk[1:]):
            counts[pair] += freq
    return counts


def _merge(seq: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the raw bytes)")
        # Work on unique chunks with counts rather than the full stream.
        chunks = Counter(tuple(m.encode("utf-8")) for m in _CHUNK_RE.findall(text))
        for new_id in range(256, vocab_size):
            counts = _pair_counts(chunks)
            if not counts:
                break
            # Tie-break on the pair itself so training is deterministic.
            best = max(counts, key=lambda p: (counts[p], (-p[0], -p[1])))
            if counts[best] < 2:
                break  # A merge seen once compresses nothing.
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            merged: Counter = Counter()
            for chunk, freq in chunks.items():
                merged[_merge(chunk, best, new_id)] += freq
            chunks = merged
        return self

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        seq = tuple(chunk)
        while len(seq) >= 2:
            pairs = set(zip(seq, seq[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            seq = _merge(seq, pair, self.merges[pair])
        return list(seq)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _CHUNK_RE.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        # One token can end mid-way through a multi-byte character, so join the
        # bytes first and decode once.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]
