"""Reference solutions. Try the problems first."""
import math
from collections import Counter

import numpy as np


def char_ngrams(word, n):
    marked = f"<{word}>"
    return [marked[i:i + n] for i in range(len(marked) - n + 1)]


def word_error_rate(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(reference)


def most_frequent_pair(ids):
    counts = Counter(zip(ids, ids[1:]))
    return min(counts, key=lambda p: (-counts[p], p))


def add_k_bigram(counts, context_total, vocab_size, word, k):
    return (counts.get(word, 0) + k) / (context_total + k * vocab_size)


def bm25_term(tf, df, n_docs, doc_len, avg_len, k1=1.5, b=0.75):
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))


def f1_from_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def viterbi(log_start, log_trans, log_emit):
    n = len(log_emit)
    V = log_start + log_emit[0]
    back = []
    for i in range(1, n):
        scores = V[:, None] + log_trans
        back.append(scores.argmax(axis=0))
        V = scores.max(axis=0) + log_emit[i]
    path = [int(V.argmax())]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    return path[::-1]


def ppmi_matrix(counts):
    expected = counts.sum(1, keepdims=True) * counts.sum(0, keepdims=True) / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(counts / expected)
    return np.maximum(np.nan_to_num(pmi, neginf=0.0, posinf=0.0), 0.0)


def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for index in np.ndindex(x.shape):
        saved = x[index]
        x[index] = saved + h
        high = f(x)
        x[index] = saved - h
        low = f(x)
        x[index] = saved
        grad[index] = (high - low) / (2 * h)
    return grad


def causal_attention(q, k, v):
    scores = q @ k.T / math.sqrt(k.shape[-1])
    scores = np.where(np.triu(np.ones(scores.shape, dtype=bool), 1), -np.inf, scores)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (exp / exp.sum(axis=-1, keepdims=True)) @ v
