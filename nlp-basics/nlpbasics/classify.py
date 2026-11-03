"""Text classification: a generative and a discriminative baseline, plus metrics.

Naive Bayes (generative) models how each class produces words:
    log P(y | doc) = log P(y) + sum_w count(w, doc) * log P(w | y) + constant
"Naive" is the assumption that words are independent given the class. It is
badly false, yet classification only needs the right argmax, so it often works.

Logistic regression (discriminative) models the decision boundary directly:
    P(y | x) = softmax(W x + b)
trained by gradient descent on cross-entropy loss, the same loss used to train
the neural models later in this course. It handles correlated features better.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Iterable, Sequence

import numpy as np


class BagOfWords:
    """Count features over unigrams and, optionally, bigrams. Bigrams give a
    linear model a cheap way to see "not good" as a unit."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 1), min_df: int = 1):
        self.ngram_range, self.min_df = ngram_range, min_df
        self.vocab: dict[str, int] = {}

    def _grams(self, tokens: Sequence[str]) -> Iterable[str]:
        low, high = self.ngram_range
        for n in range(low, high + 1):
            for i in range(len(tokens) - n + 1):
                yield " ".join(tokens[i:i + n])

    def fit(self, docs: Sequence[Sequence[str]]) -> "BagOfWords":
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(self._grams(doc)))
        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(kept)}
        return self

    def transform(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.vocab)))
        for row, doc in enumerate(docs):
            for gram in self._grams(doc):
                col = self.vocab.get(gram)
                if col is not None:
                    X[row, col] += 1
        return X

    def fit_transform(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        return self.fit(docs).transform(docs)

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.vocab, key=self.vocab.get)


class MultinomialNB:
    """alpha is Laplace smoothing, the same idea as in n-gram models. Without
    it, one unseen word in a class zeroes that class entirely."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: Sequence[Hashable]) -> "MultinomialNB":
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        counts = np.stack([X[y == c].sum(axis=0) for c in self.classes_])
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log([(y == c).mean() for c in self.classes_])
        return self

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]

    def most_indicative(self, names: Sequence[str], top: int = 8) -> dict:
        """Top features by log-odds between classes: P(w|c) / P(w|other)."""
        result = {}
        for i, c in enumerate(self.classes_):
            others = np.delete(self.feature_log_prob_, i, axis=0).max(axis=0)
            ratios = self.feature_log_prob_[i] - others
            result[c.item()] = [(names[j], float(ratios[j]))
                                for j in np.argsort(-ratios)[:top]]
        return result


class LogisticRegression:
    """Softmax regression by full-batch gradient descent with momentum.

    Gradient of mean cross-entropy, written out so nothing is hidden:
        dL/dW = X^T (p - Y) / n + l2 * W
    """

    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3,
                 momentum: float = 0.9):
        self.lr, self.epochs, self.l2, self.momentum = lr, epochs, l2, momentum
        self.losses: list[float] = []

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        # Subtracting the row max before exp prevents overflow and leaves the
        # result unchanged, since softmax ignores a constant shift.
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: Sequence[Hashable]) -> "LogisticRegression":
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        index = {c: i for i, c in enumerate(self.classes_.tolist())}
        n, d, k = X.shape[0], X.shape[1], len(self.classes_)
        Y = np.zeros((n, k))
        Y[np.arange(n), [index[c] for c in y.tolist()]] = 1
        self.W, self.b = np.zeros((d, k)), np.zeros(k)
        vW, vb = np.zeros_like(self.W), np.zeros_like(self.b)
        self.losses = []
        for _ in range(self.epochs):
            p = self._softmax(X @ self.W + self.b)
            loss = -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))
            self.losses.append(float(loss + 0.5 * self.l2 * np.sum(self.W ** 2)))
            error = p - Y
            vW = self.momentum * vW - self.lr * (X.T @ error / n + self.l2 * self.W)
            vb = self.momentum * vb - self.lr * error.mean(axis=0)
            self.W, self.b = self.W + vW, self.b + vb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def train_test_split(items: Sequence, labels: Sequence, test_size: float = 0.25,
                     seed: int = 0, groups: Sequence | None = None) -> tuple[list, list, list, list]:
    """Stratified split: each class keeps its proportion in both parts.

    With groups given, items sharing a group id stay on the same side instead.
    Use this whenever items are related: near-duplicates, sentences from the
    same document, or messages by the same author.
    """
    rng = random.Random(seed)
    train_idx, test_idx = [], []
    if groups is not None:
        ids = sorted(set(groups), key=str)
        rng.shuffle(ids)
        held = set(ids[:max(1, round(len(ids) * test_size))])
        train_idx = [i for i, g in enumerate(groups) if g not in held]
        test_idx = [i for i, g in enumerate(groups) if g in held]
        labels_for_stratifying: Sequence = ()
    else:
        labels_for_stratifying = labels
    for label in sorted(set(labels_for_stratifying), key=str):
        idx = [i for i, l in enumerate(labels) if l == label]
        rng.shuffle(idx)
        cut = max(1, round(len(idx) * test_size))
        test_idx += idx[:cut]
        train_idx += idx[cut:]
    pick = lambda seq, ids: [seq[i] for i in ids]
    return pick(items, train_idx), pick(items, test_idx), pick(labels, train_idx), pick(labels, test_idx)


def confusion_matrix(y_true: Sequence, y_pred: Sequence) -> tuple[list, np.ndarray]:
    labels = sorted(set(y_true) | set(y_pred), key=str)
    index = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return labels, matrix


def classification_report(y_true: Sequence, y_pred: Sequence) -> dict:
    """Precision: of what I flagged, how much was right?  Recall: of what was
    there, how much did I find?  F1 is their harmonic mean. Accuracy alone is
    misleading on imbalanced data: always predicting "not spam" on 99% ham
    scores 99% while finding no spam."""
    labels, m = confusion_matrix(y_true, y_pred)
    report = {}
    for i, label in enumerate(labels):
        tp, fp, fn = m[i, i], m[:, i].sum() - m[i, i], m[i, :].sum() - m[i, i]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": float(precision), "recall": float(recall),
                         "f1": float(f1), "support": int(m[i].sum())}
    report["accuracy"] = float(np.trace(m) / m.sum())
    report["macro_f1"] = float(np.mean([report[l]["f1"] for l in labels]))
    return report


def cross_val_accuracy(model_factory, docs: Sequence, labels: Sequence, folds: int = 5,
                       seed: int = 0, groups: Sequence | None = None) -> list[float]:
    """k-fold cross-validation. On small data, one split's score is mostly luck;
    the spread across folds tells you how much to trust it.

    model_factory() must return a fresh (vectorizer, classifier) pair, and the
    vectorizer is fit inside each fold. Fitting on all data first would leak
    test vocabulary into training. With groups given, a group is never divided
    between train and test.
    """
    order = list(range(len(docs)))
    rng = random.Random(seed)
    rng.shuffle(order)
    group_ids = sorted(set(groups), key=str) if groups is not None else []
    rng.shuffle(group_ids)
    scores = []
    for fold in range(folds):
        if groups is not None:
            held = set(group_ids[fold::folds])
            test_ids = {i for i in order if groups[i] in held}
        else:
            test_ids = set(order[fold::folds])
        tr = [i for i in order if i not in test_ids]
        te = [i for i in order if i in test_ids]
        vec, clf = model_factory()
        clf.fit(vec.fit_transform([docs[i] for i in tr]), [labels[i] for i in tr])
        predicted = clf.predict(vec.transform([docs[i] for i in te]))
        scores.append(float(np.mean([p == labels[i] for p, i in zip(predicted.tolist(), te)])))
    return scores
