"""Hidden Markov Model part-of-speech tagging with the Viterbi algorithm.

Sequence labeling means choosing one tag per token jointly. "book" is a noun
after "the" and a verb after "will", so tags cannot be picked independently.

An HMM tells a generative story: hidden tags form a Markov chain, and each tag
emits a word.

    P(words, tags) = prod_i P(tag_i | tag_i-1) * P(word_i | tag_i)
                              transition          emission

With T tags and n words there are T^n tag sequences, but Viterbi finds the best
in O(n T^2) by dynamic programming. The best path to tag t at position i must
extend the best path to SOME tag at i-1, so only those need remembering:

    V[i][t] = max_s V[i-1][s] + log P(t | s) + log P(word_i | t)

Replace max with sum and you get the forward algorithm, which computes the total
probability of the words across every possible tag sequence.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

START, END = "<START>", "<END>"
TaggedSentence = Sequence[tuple[str, str]]


def word_shape(word: str) -> str:
    """Class for unknown words. Suffixes and capitalization carry real signal,
    so a word never seen before can still look like a verb or a number."""
    if any(c.isdigit() for c in word):
        return "<UNK-NUM>"
    if word[0].isupper():
        return "<UNK-CAP>"
    for suffix in ("ing", "ed", "ly", "tion", "ness", "er", "est", "s"):
        if word.endswith(suffix):
            return f"<UNK-{suffix}>"
    return "<UNK>"


class HMMTagger:
    def __init__(self, emission_k: float = 0.01, transition_k: float = 0.1,
                 use_shapes: bool = True):
        self.emission_k, self.transition_k = emission_k, transition_k
        self.use_shapes = use_shapes

    def _map_word(self, word: str) -> str:
        if word in self.known:
            return word
        return word_shape(word) if self.use_shapes else "<UNK>"

    def fit(self, sentences: Sequence[TaggedSentence]) -> "HMMTagger":
        freq = Counter(w for s in sentences for w, _ in s)
        # Words seen once stand in for unknown words: they are the best sample
        # we have of what an unseen word looks like. Good-Turing's insight.
        self.known = {w for w, c in freq.items() if c > 1}
        transitions: dict[str, Counter] = defaultdict(Counter)
        emissions: dict[str, Counter] = defaultdict(Counter)
        for sentence in sentences:
            previous = START
            for word, tag in sentence:
                transitions[previous][tag] += 1
                emissions[tag][self._map_word(word)] += 1
                previous = tag
            transitions[previous][END] += 1

        self.tags = sorted(emissions)
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        self.words = sorted({w for c in emissions.values() for w in c})
        self.word_index = {w: i for i, w in enumerate(self.words)}
        T, V = len(self.tags), len(self.words)

        def smoothed_row(counter: Counter, keys: list[str], k: float) -> np.ndarray:
            row = np.array([counter[key] + k for key in keys], dtype=float)
            return np.log(row / row.sum())

        self.log_start = smoothed_row(transitions[START], self.tags, self.transition_k)
        self.log_end = np.array([
            smoothed_row(transitions[t], self.tags + [END], self.transition_k)[-1]
            for t in self.tags])
        self.log_trans = np.stack([
            smoothed_row(transitions[t], self.tags + [END], self.transition_k)[:-1]
            for t in self.tags])
        self.log_emit = np.stack([
            smoothed_row(emissions[t], self.words, self.emission_k) for t in self.tags])
        self.log_unseen = np.log(np.array([
            self.emission_k / (sum(emissions[t].values()) + self.emission_k * (V + 1))
            for t in self.tags]))
        assert self.log_trans.shape == (T, T)
        return self

    def _emission(self, word: str) -> np.ndarray:
        index = self.word_index.get(self._map_word(word))
        return self.log_unseen if index is None else self.log_emit[:, index]

    def viterbi(self, words: Sequence[str]) -> tuple[list[str], float]:
        """Return the most probable tag sequence and its joint log-probability."""
        n, T = len(words), len(self.tags)
        if n == 0:
            return [], 0.0
        V = np.full((n, T), -np.inf)
        back = np.zeros((n, T), dtype=int)
        V[0] = self.log_start + self._emission(words[0])
        for i in range(1, n):
            # scores[s, t] is the best path ending in s, then the step s -> t.
            scores = V[i - 1][:, None] + self.log_trans
            back[i] = scores.argmax(axis=0)
            V[i] = scores.max(axis=0) + self._emission(words[i])
        final = V[-1] + self.log_end
        state = int(final.argmax())
        best = float(final[state])
        path = [state]
        for i in range(n - 1, 0, -1):
            state = int(back[i, state])
            path.append(state)
        return [self.tags[s] for s in reversed(path)], best

    def forward(self, words: Sequence[str]) -> float:
        """log P(words), summed over all tag sequences. Identical to Viterbi with
        max replaced by log-sum-exp."""
        alpha = self.log_start + self._emission(words[0])
        for word in words[1:]:
            alpha = np.logaddexp.reduce(alpha[:, None] + self.log_trans, axis=0) + self._emission(word)
        return float(np.logaddexp.reduce(alpha + self.log_end))

    def joint_logprob(self, words: Sequence[str], tags: Sequence[str]) -> float:
        """log P(words, tags) for one specific sequence. Used by the tests to
        confirm Viterbi against brute-force search."""
        ids = [self.tag_index[t] for t in tags]
        total = self.log_start[ids[0]] + self._emission(words[0])[ids[0]]
        for i in range(1, len(ids)):
            total += self.log_trans[ids[i - 1], ids[i]] + self._emission(words[i])[ids[i]]
        return float(total + self.log_end[ids[-1]])


class MostFrequentTagBaseline:
    """Tag each word with its most common training tag, ignoring context. Always
    compute a baseline like this: it tells you how much the sequence model adds."""

    def fit(self, sentences: Sequence[TaggedSentence]) -> "MostFrequentTagBaseline":
        counts: dict[str, Counter] = defaultdict(Counter)
        tag_totals: Counter = Counter()
        for sentence in sentences:
            for word, tag in sentence:
                counts[word][tag] += 1
                tag_totals[tag] += 1
        self.table = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        self.default = tag_totals.most_common(1)[0][0]
        return self

    def tag(self, words: Sequence[str]) -> list[str]:
        return [self.table.get(w, self.default) for w in words]


def tagging_accuracy(predict, sentences: Sequence[TaggedSentence],
                     known: set[str] | None = None) -> dict[str, float]:
    """Token accuracy overall, optionally split into known and unknown words."""
    right = wrong = oov_right = oov_total = 0
    for sentence in sentences:
        words = [w for w, _ in sentence]
        for (word, gold), guess in zip(sentence, predict(words)):
            right += gold == guess
            wrong += gold != guess
            if known is not None and word not in known:
                oov_total += 1
                oov_right += gold == guess
    result = {"accuracy": right / (right + wrong)}
    if known is not None:
        result["oov_accuracy"] = oov_right / oov_total if oov_total else math.nan
        result["oov_tokens"] = oov_total
    return result
