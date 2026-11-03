"""Word embeddings: "You shall know a word by the company it keeps" (Firth, 1957).

One-hot vectors make every pair of words equally unrelated. The distributional
hypothesis says words in similar contexts have similar meanings, so we derive
dense vectors from co-occurrence. Two routes, one idea:

  Count-based: build a co-occurrence matrix, reweight it with PPMI, compress it
  with truncated SVD. This is latent semantic analysis, modernized.

  Prediction-based: skip-gram with negative sampling (word2vec). Train a
  classifier to separate real (word, context) pairs from random ones, and keep
  its weights as the embeddings.

These are closer than they look: Levy & Goldberg (2014) showed that skip-gram
with negative sampling implicitly factorizes a shifted PMI matrix.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def cooccurrence_matrix(sentences: Sequence[Sequence[str]], window: int = 2,
                        min_count: int = 1) -> tuple[np.ndarray, list[str]]:
    freq = Counter(w for s in sentences for w in s)
    vocab = sorted(w for w, c in freq.items() if c >= min_count)
    index = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for sentence in sentences:
        ids = [index[w] for w in sentence if w in index]
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i != j:
                    C[center, ids[j]] += 1
    return C, vocab


def ppmi(C: np.ndarray) -> np.ndarray:
    """Positive pointwise mutual information:

        PMI(w, c) = log( P(w, c) / (P(w) P(c)) )

    How much more often do w and c co-occur than chance predicts? "the" appears
    near everything, so raw counts overrate it; PMI corrects this. Negative
    values are unreliable estimates on sparse data, so they are clipped to 0.
    """
    total = C.sum()
    row, col = C.sum(axis=1, keepdims=True), C.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C * total / (row @ col))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


def svd_embeddings(M: np.ndarray, dim: int = 25) -> np.ndarray:
    """Truncated SVD gives the best rank-k approximation of M (Eckart-Young),
    which forces words with similar context rows onto similar short vectors.
    Scaling by sqrt(S) rather than S is a common empirical choice."""
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :dim] * np.sqrt(S[:dim])


class SkipGram:
    """Skip-gram with negative sampling, gradients derived by hand.

    For a true pair (w, c) and k noise words n_i, maximize

        log sigma(v_w . u_c) + sum_i log sigma(-v_w . u_ni)

    Differentiating gives simple updates:
        du_c = (sigma(v.u) - 1) * v           pull true context closer
        du_n =  sigma(v.u_n)    * v           push noise away
        dv   = sum of the matching terms in u

    Noise words are drawn from unigram^0.75, which raises rare words' odds
    relative to frequent ones. Frequent words are randomly dropped from the
    training stream with Mikolov's subsampling heuristic.
    """

    def __init__(self, dim: int = 30, window: int = 3, negatives: int = 5, lr: float = 0.05,
                 epochs: int = 5, min_count: int = 1, subsample: float = 1e-3, seed: int = 0):
        self.dim, self.window, self.negatives = dim, window, negatives
        self.lr, self.epochs, self.min_count = lr, epochs, min_count
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)
        self.losses: list[float] = []

    def fit(self, sentences: Sequence[Sequence[str]]) -> "SkipGram":
        freq = Counter(w for s in sentences for w in s)
        self.vocab = sorted(w for w, c in freq.items() if c >= self.min_count)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        counts = np.array([freq[w] for w in self.vocab], dtype=float)
        noise = counts ** 0.75
        noise /= noise.sum()
        fraction = counts / counts.sum()
        keep = np.minimum(1.0, np.sqrt(self.subsample / fraction) + self.subsample / fraction)

        V = len(self.vocab)
        self.W = (self.rng.random((V, self.dim)) - 0.5) / self.dim
        self.U = np.zeros((V, self.dim))
        encoded = [np.array([self.index[w] for w in s if w in self.index], dtype=int)
                   for s in sentences]
        total_steps = max(1, self.epochs * sum(len(s) for s in encoded))
        step = 0
        for _ in range(self.epochs):
            epoch_loss, pairs = 0.0, 0
            for s in encoded:
                kept = s[self.rng.random(len(s)) < keep[s]]
                for i, center in enumerate(kept):
                    step += 1
                    # A random window size weights nearer words more heavily.
                    w = int(self.rng.integers(1, self.window + 1))
                    lr = self.lr * max(1 - step / total_steps, 1e-3)
                    for j in range(max(0, i - w), min(len(kept), i + w + 1)):
                        if j != i:
                            epoch_loss += self._update(center, kept[j], noise, lr)
                            pairs += 1
            self.losses.append(epoch_loss / max(pairs, 1))
        return self

    def _update(self, center: int, context: int, noise: np.ndarray, lr: float) -> float:
        targets = np.concatenate(([context], self.rng.choice(len(noise), self.negatives, p=noise)))
        labels = np.zeros(len(targets))
        labels[0] = 1.0
        v = self.W[center]
        scores = np.clip(self.U[targets] @ v, -10, 10)
        sigma = 1.0 / (1.0 + np.exp(-scores))
        error = sigma - labels
        grad_v = error @ self.U[targets]       # Uses U before this step's update.
        self.U[targets] -= lr * error[:, None] * v
        self.W[center] -= lr * grad_v
        return float(-np.log(sigma[0] + 1e-10) - np.log(1 - sigma[1:] + 1e-10).sum())


class KeyedVectors:
    """Query helper. Vectors are unit-normalized, so a dot product is a cosine."""

    def __init__(self, vectors: np.ndarray, vocab: Sequence[str]):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.where(norms == 0, 1, norms)
        self.vocab, self.index = list(vocab), {w: i for i, w in enumerate(vocab)}

    def similarity(self, a: str, b: str) -> float:
        return float(self.vectors[self.index[a]] @ self.vectors[self.index[b]])

    def nearest(self, query: str | np.ndarray, top: int = 5,
                exclude: Sequence[str] = ()) -> list[tuple[str, float]]:
        vector = self.vectors[self.index[query]] if isinstance(query, str) else query
        skip = set(exclude) | ({query} if isinstance(query, str) else set())
        scores = self.vectors @ vector
        ranked = (i for i in np.argsort(-scores) if self.vocab[i] not in skip)
        return [(self.vocab[i], float(scores[i])) for _, i in zip(range(top), ranked)]

    def analogy(self, a: str, b: str, c: str, top: int = 1) -> list[tuple[str, float]]:
        """a is to b as c is to ?  via 3CosAdd: nearest(b - a + c). Excluding the
        inputs is standard, and matters: without it the top answer is often an
        input word, a caveat behind many famous analogy demos."""
        query = (self.vectors[self.index[b]] - self.vectors[self.index[a]]
                 + self.vectors[self.index[c]])
        return self.nearest(query / np.linalg.norm(query), top=top, exclude=(a, b, c))
