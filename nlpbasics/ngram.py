"""N-gram language models: the first probabilistic model of language.

Chain rule, exactly:   P(w1..wn) = prod_i P(wi | w1..w(i-1))
Markov approximation:  P(wi | w1..w(i-1)) ~= P(wi | w(i-n+1)..w(i-1))
Maximum likelihood:    P(w | ctx) = count(ctx, w) / count(ctx)

The maximum likelihood estimate assigns probability zero to anything unseen,
and one zero makes a whole sentence's probability zero. Smoothing is the fix:
move a little probability mass from seen events to unseen ones.

Perplexity is the standard metric: the geometric-mean inverse probability per
token. It reads as an effective branching factor, "how many equally likely next
words does this model feel it is choosing between?" Lower is better.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Sequence

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramModel:
    """Order-n model with one of three estimators:

      "mle"            raw counts; infinite perplexity on unseen n-grams
      "laplace"        add-k: pretend every n-gram was seen k extra times
      "interpolation"  mix all orders: l3*P(w|uv) + l2*P(w|v) + l1*P(w) + ...
    """

    def __init__(self, n: int = 2, smoothing: str = "interpolation", k: float = 1.0,
                 lambdas: Sequence[float] | None = None, min_count: int = 1):
        if smoothing not in {"mle", "laplace", "interpolation"}:
            raise ValueError(f"unknown smoothing: {smoothing}")
        self.n, self.smoothing, self.k, self.min_count = n, smoothing, k, min_count
        # lambdas[0] weights the highest order. Default favors specific contexts.
        self.lambdas = list(lambdas) if lambdas else self._default_lambdas(n)
        if not math.isclose(sum(self.lambdas), 1.0):
            raise ValueError("lambdas must sum to 1")
        self.counts: list[Counter] = []   # counts[o][(w1..w_o+1)]
        self.context_totals: list[Counter] = []
        self.next_words: dict[tuple, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    @staticmethod
    def _default_lambdas(n: int) -> list[float]:
        weights = [2.0 ** (n - i) for i in range(n)]
        return [w / sum(weights) for w in weights]

    def _pad(self, sentence: Sequence[str]) -> list[str]:
        known = [w if w in self.vocab else UNK for w in sentence]
        return [BOS] * (self.n - 1) + known + [EOS]

    def fit(self, sentences: Iterable[Sequence[str]]) -> "NGramModel":
        sentences = [list(s) for s in sentences]
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= self.min_count} | {EOS, UNK}
        self.counts = [Counter() for _ in range(self.n)]
        self.context_totals = [Counter() for _ in range(self.n)]
        for sentence in sentences:
            padded = self._pad(sentence)
            for i in range(self.n - 1, len(padded)):
                for order in range(self.n):
                    gram = tuple(padded[i - order:i + 1])
                    self.counts[order][gram] += 1
                    self.context_totals[order][gram[:-1]] += 1
                self.next_words[tuple(padded[i - self.n + 1:i])][padded[i]] += 1
        return self

    def _mle(self, word: str, context: tuple[str, ...]) -> float:
        order = len(context)
        total = self.context_totals[order][context]
        return self.counts[order][context + (word,)] / total if total else 0.0

    def prob(self, word: str, context: Sequence[str]) -> float:
        """P(word | context). The context is cut to the last n-1 tokens."""
        word = word if word in self.vocab else UNK
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        if self.smoothing == "mle":
            return self._mle(word, context)
        if self.smoothing == "laplace":
            order = len(context)
            numerator = self.counts[order][context + (word,)] + self.k
            return numerator / (self.context_totals[order][context] + self.k * len(self.vocab))
        # Interpolation. An unseen context contributes nothing at that order, so
        # its weight is redistributed in proportion among orders that remain.
        # The uniform floor guarantees every token keeps nonzero probability.
        total, weight_used = 0.0, 0.0
        for i, lam in enumerate(self.lambdas):
            ctx = context[i:] if i < len(context) else ()
            if self.context_totals[len(ctx)][ctx]:
                total += lam * self._mle(word, ctx)
                weight_used += lam
        estimate = total / weight_used if weight_used else 0.0
        return 0.99 * estimate + 0.01 / len(self.vocab)

    def logprob(self, sentence: Sequence[str]) -> float:
        """Sum of logs, not product of probabilities: a 100-word sentence would
        underflow a float to exactly 0.0."""
        padded = self._pad(sentence)
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            p = self.prob(padded[i], padded[i - self.n + 1:i] if self.n > 1 else ())
            if p == 0.0:
                return -math.inf
            total += math.log(p)
        return total

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """exp(-average log-likelihood per token). Must use held-out text: on
        training data, a bigger model always looks better."""
        log_likelihood, tokens = 0.0, 0
        for sentence in sentences:
            log_likelihood += self.logprob(sentence)
            tokens += len(sentence) + 1   # +1 for </s>
        return math.exp(-log_likelihood / tokens) if math.isfinite(log_likelihood) else math.inf

    def generate(self, max_len: int = 30, seed: int | None = None,
                 prefix: Sequence[str] = ()) -> list[str]:
        """Sample one token at a time from the raw counts of the context seen,
        backing off to shorter contexts when the current one is unknown."""
        rng = random.Random(seed)
        out = [BOS] * (self.n - 1) + list(prefix)
        for _ in range(max_len):
            ctx = tuple(out[-(self.n - 1):]) if self.n > 1 else ()
            options = self.next_words.get(ctx)
            if options is None:
                options = self.counts[0]
                words, weights = [g[0] for g in options], list(options.values())
            else:
                words, weights = list(options), list(options.values())
            word = rng.choices(words, weights=weights)[0]
            if word == EOS:
                break
            out.append(word)
        return out[self.n - 1:]


def grid_search_lambdas(train: Sequence[Sequence[str]], heldout: Sequence[Sequence[str]],
                        n: int = 3, step: float = 0.1,
                        min_count: int = 2) -> tuple[list[float], float]:
    """Choose hyperparameters on a held-out DEV set, never on the test set.
    Returns (best lambdas, dev perplexity) for a trigram model."""
    if n != 3:
        raise ValueError("grid search is implemented for trigrams")
    base = NGramModel(3, "interpolation", min_count=min_count).fit(train)
    best, best_score = base.lambdas, base.perplexity(heldout)
    steps = int(round(1 / step))
    for a in range(1, steps):
        for b in range(1, steps - a):
            base.lambdas = [a * step, b * step, 1 - (a + b) * step]
            score = base.perplexity(heldout)
            if score < best_score:
                best, best_score = list(base.lambdas), score
    base.lambdas = best
    return best, best_score
