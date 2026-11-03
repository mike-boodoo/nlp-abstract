"""Edit distance and spelling correction: dynamic programming meets probability.

D[i][j] = minimum edits turning the first i chars of a into the first j of b.

    D[i][j] = min( D[i-1][j]   + 1,          delete a[i-1]
                   D[i][j-1]   + 1,          insert b[j-1]
                   D[i-1][j-1] + (0 or 1) )  match or substitute

Same recurrence shape as Viterbi (lesson 7): best solution to a prefix, built
from best solutions to smaller prefixes.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def edit_matrix(a: Sequence, b: Sequence) -> list[list[int]]:
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
    return D


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Works on any sequences. On lists of words it gives word error rate, the
    standard metric for speech recognition."""
    return edit_matrix(a, b)[len(a)][len(b)]


def damerau_levenshtein(a: Sequence, b: Sequence) -> int:
    """Optimal string alignment: also counts swapping two adjacent characters as
    one edit, because "teh" for "the" is a single human slip."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
    return D[n][m]


def align(a: Sequence, b: Sequence) -> list[tuple[str, object, object]]:
    """Backtrace through the matrix to recover the actual edit script as a list
    of (op, from, to), with op one of "=", "sub", "del", "ins"."""
    D = edit_matrix(a, b)
    i, j, ops = len(a), len(b), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            ops.append(("=", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    return ops[::-1]


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def edits1(word: str) -> set[str]:
    """All strings one edit away. For length n and alphabet size 26 that is
    about 54n + 25 candidates, which is why this stays cheap."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l, r in splits if r]
    swaps = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in ALPHABET]
    inserts = [l + c + r for l, r in splits for c in ALPHABET]
    return set(deletes + swaps + replaces + inserts)


class SpellingCorrector:
    """Noisy channel model, the classic formulation of correction:

        best = argmax_c  P(c) * P(typed | c)

    P(c) is the language model: how common is this word?
    P(typed | c) is the error model: how plausibly does c get typed this way?

    This also explains why distance alone is not enough: "thw" is one edit from
    both "the" and "thaw", and the prior breaks the tie toward "the". Here the
    error model is deliberately crude, a fixed penalty per edit.
    """

    def __init__(self, words: Iterable[str], edit_penalty: float = 1e-3):
        self.counts = Counter(w.lower() for w in words)
        self.total = sum(self.counts.values())
        self.log_penalty = math.log(edit_penalty)

    def log_prior(self, word: str) -> float:
        return math.log(self.counts[word] / self.total)

    def candidates(self, word: str) -> dict[str, float]:
        """Map each known candidate to its log score."""
        word = word.lower()
        if word in self.counts:
            return {word: self.log_prior(word)}
        one = {w for w in edits1(word) if w in self.counts}
        scored = {w: self.log_prior(w) + self.log_penalty for w in one}
        if not scored:
            two = {e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in self.counts}
            scored = {w: self.log_prior(w) + 2 * self.log_penalty for w in two}
        return scored

    def correct(self, word: str) -> str:
        scored = self.candidates(word)
        return max(sorted(scored), key=scored.get) if scored else word
