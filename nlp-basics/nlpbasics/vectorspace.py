"""The vector space model: a document is a point, similarity is geometry.

Bag of words throws away order and keeps counts. Term frequency alone is
dominated by common words, so TF-IDF multiplies by rarity:

    tfidf(t, d) = tf(t, d) * ( ln((1 + N) / (1 + df(t))) + 1 )

A word in every document gets the minimum weight; a word in few documents
gets a large one. This smoothed form matches scikit-learn's default, and the
test suite checks agreement to numerical precision.

BM25 improves ranking with two ideas TF-IDF lacks:
  saturation  the 10th occurrence of a term adds less than the 2nd (k1)
  length      long documents are penalized for matching by sheer size (b)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Sequence

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle: dot product over norms. It ignores vector length, so
    a document and that document repeated twice score 1.0."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


class TfidfVectorizer:
    def __init__(self, tokenizer: Callable[[str], list[str]] = str.split):
        self.tokenizer = tokenizer
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def fit(self, documents: Sequence[str]) -> "TfidfVectorizer":
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(self.tokenizer(doc)))      # set: documents, not occurrences
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(documents)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.vocab)))
        for row, doc in enumerate(documents):
            for token, count in Counter(self.tokenizer(doc)).items():
                if token in self.vocab:
                    matrix[row, self.vocab[token]] = count
        matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


class BM25:
    """Okapi BM25 over an inverted index, the structure behind search engines:
    term -> [(doc id, term count), ...]. Querying touches only documents that
    contain a query term, not the entire collection.

        score(q, d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
        idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
    """

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in documents]
        self.avgdl = sum(self.lengths) / len(documents)
        self.n = len(documents)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                self.postings[term].append((doc_id, tf))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: Sequence[str], top_k: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in query:
            idf = self.idf(term)
            for doc_id, tf in self.postings.get(term, ()):
                length_norm = 1 - self.b + self.b * self.lengths[doc_id] / self.avgdl
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def average_precision(ranked: Sequence[int], relevant: set[int]) -> float:
    """Mean of precision@k at each rank k that holds a relevant document. Rewards
    placing relevant documents early."""
    hits, total = 0, 0.0
    for rank, doc in enumerate(ranked, 1):
        if doc in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant) if relevant else 0.0
