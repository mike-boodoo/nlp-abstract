"""Auto-checker. Each check calls the real nlpbasics code where one exists."""
import math
import sys

import numpy as np

from nlpbasics import distance, embeddings, hmm, neural, vectorspace
from nlpbasics.autograd import Tensor


def check_char_ngrams(m):
    assert m.char_ngrams("cat", 3) == ["<ca", "cat", "at>"]
    assert m.char_ngrams("a", 3) == ["<a>"]


def check_wer(m):
    ref, hyp = "the cat sat on the mat".split(), "the cat sat on mat today".split()
    assert math.isclose(m.word_error_rate(ref, hyp), distance.levenshtein(ref, hyp) / 6)
    assert m.word_error_rate(["a"], ["a"]) == 0


def check_pair(m):
    assert m.most_frequent_pair([1, 2, 3, 1, 2]) == (1, 2)
    assert m.most_frequent_pair([5, 6, 1, 2]) == (1, 2)


def check_add_k(m):
    counts = {"cat": 3, "dog": 1}
    assert math.isclose(m.add_k_bigram(counts, 4, 10, "cat", 1), 4 / 14)
    assert math.isclose(sum(m.add_k_bigram(counts, 4, 3, w, .5) for w in ("cat", "dog", "emu")), 1)


def check_bm25(m):
    docs = [["a", "b", "a"], ["b", "c"], ["c", "c", "d", "e"]]
    index = vectorspace.BM25(docs)
    score = dict(index.search(["a"]))[0]
    assert math.isclose(m.bm25_term(2, 1, 3, 3, 3.0), score)


def check_f1(m):
    assert math.isclose(m.f1_from_counts(8, 2, 2), 0.8)
    assert m.f1_from_counts(0, 0, 0) == 0.0


def check_viterbi(m):
    train = [[("a", "X"), ("b", "Y")], [("a", "X"), ("a", "Y")], [("b", "Y"), ("a", "X")]] * 3
    tagger = hmm.HMMTagger(use_shapes=False).fit(train)
    words = ["a", "b", "a", "a"]
    emit = np.stack([tagger._emission(w) for w in words])
    path = m.viterbi(tagger.log_start, tagger.log_trans, emit)
    no_end = hmm.HMMTagger(use_shapes=False).fit(train)
    no_end.log_end = np.zeros_like(no_end.log_end)
    assert [no_end.tags[i] for i in path] == no_end.viterbi(words)[0]


def check_ppmi(m):
    counts = np.array([[4.0, 0, 1], [0, 2, 2], [1, 2, 0]])
    assert np.allclose(m.ppmi_matrix(counts), embeddings.ppmi(counts))


def check_gradient(m):
    x = np.array([1.0, 2.0, -1.0])
    grad = m.numerical_gradient(lambda v: float((v ** 3).sum()), x)
    assert np.allclose(grad, 3 * x ** 2, atol=1e-6)
    assert np.array_equal(x, [1.0, 2.0, -1.0])


def check_attention(m):
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(3, 5, 4))
    mask = np.triu(np.ones((5, 5), dtype=bool), 1)
    expected, _ = neural.scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask)
    assert np.allclose(m.causal_attention(q, k, v), expected.data, atol=1e-6)


CHECKS = [check_char_ngrams, check_wer, check_pair, check_add_k, check_bm25, check_f1,
          check_viterbi, check_ppmi, check_gradient, check_attention]


def main() -> int:
    if "--solutions" in sys.argv:
        from . import solutions as module
    else:
        from . import problems as module
    passed = 0
    for number, check in enumerate(CHECKS, 1):
        name = check.__name__.removeprefix("check_")
        try:
            check(module)
            print(f"[pass] {number:>2}. {name}")
            passed += 1
        except NotImplementedError:
            print(f"[todo] {number:>2}. {name}")
        except Exception as error:
            print(f"[FAIL] {number:>2}. {name}: {type(error).__name__} {error}")
    print(f"\n{passed}/{len(CHECKS)} passed")
    return 0 if "--solutions" not in sys.argv or passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
