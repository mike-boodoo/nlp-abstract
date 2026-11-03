"""Exercises, one per lesson. Replace each NotImplementedError, then run:

    python -m exercises.check            check your work
    python -m exercises.check --solutions  confirm the checker against the answers

Everything can be done with the standard library and NumPy. Do not import from
nlpbasics here: the point is to rewrite the core idea yourself.
"""
import numpy as np


def char_ngrams(word: str, n: int) -> list[str]:
    """Lesson 1. Character n-grams with boundary markers, as in fastText.
    char_ngrams("cat", 3) == ["<ca", "cat", "at>"]"""
    raise NotImplementedError


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Lesson 2. Word-level edit distance divided by len(reference)."""
    raise NotImplementedError


def most_frequent_pair(ids: list[int]) -> tuple[int, int]:
    """Lesson 3. Most common adjacent pair. Break ties by the smallest pair."""
    raise NotImplementedError


def add_k_bigram(counts: dict, context_total: int, vocab_size: int, word: str, k: float) -> float:
    """Lesson 4. Add-k estimate of P(word | context), where counts maps each word
    to its count after this context and context_total is the count of the context."""
    raise NotImplementedError


def bm25_term(tf: int, df: int, n_docs: int, doc_len: int, avg_len: float,
              k1: float = 1.5, b: float = 0.75) -> float:
    """Lesson 5. BM25 score contribution of one term in one document."""
    raise NotImplementedError


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Lesson 6. F1 score. Return 0.0 when it is undefined."""
    raise NotImplementedError


def viterbi(log_start: np.ndarray, log_trans: np.ndarray, log_emit: np.ndarray) -> list[int]:
    """Lesson 7. log_start has shape (T,), log_trans (T, T), and log_emit (n, T):
    the emission score of each state at each position. Return the best path."""
    raise NotImplementedError


def ppmi_matrix(counts: np.ndarray) -> np.ndarray:
    """Lesson 8. Positive PMI of a co-occurrence matrix. Zero counts map to 0."""
    raise NotImplementedError


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Lesson 9. Central-difference gradient of scalar f at x. Do not leave x changed."""
    raise NotImplementedError


def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Lesson 10. Scaled dot-product self-attention with a causal mask, all inputs
    of shape (T, d). Each position t may attend only to positions 0..t."""
    raise NotImplementedError
