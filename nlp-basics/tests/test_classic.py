"""Text, stemming, distance, BPE, n-grams, retrieval, classification, tagging."""
import itertools
import math
import random

import numpy as np
import pytest

from nlpbasics import bpe, classify, data, distance, embeddings, hmm, ngram, stem, text, vectorspace


def test_normalize_unifies_composed_and_decomposed():
    assert text.normalize("café") == text.normalize("café")


def test_tokenizer_keeps_units_together():
    tokens = text.tokenize("Pay $1,000.50 to the U.S. office, see https://x.org/a?b=1).")
    assert {"$1,000.50", "U.S.", "https://x.org/a?b=1"} <= set(tokens)
    assert tokens[-2:] == [")", "."]


def test_contractions():
    assert text.split_contractions(["didn't", "she's", "cat"]) == ["did", "n't", "she", "'s", "cat"]


def test_sentence_splitter_respects_abbreviations_and_decimals():
    assert len(text.split_sentences("Dr. Lee paid 3.5 dollars. He left. Did she?")) == 3


def test_vocabulary_round_trip_and_unknowns():
    v = text.Vocabulary(min_freq=2).fit([["a", "a", "b"]])
    assert v.decode(v.encode(["a", "b"])) == ["a", "<unk>"]


# Examples printed in Porter (1980).
@pytest.mark.parametrize("word,expected", [
    ("caresses", "caress"), ("ponies", "poni"), ("cats", "cat"), ("feed", "feed"),
    ("agreed", "agre"), ("plastered", "plaster"), ("motoring", "motor"), ("sing", "sing"),
    ("conflated", "conflat"), ("hopping", "hop"), ("filing", "file"), ("happy", "happi"),
    ("relational", "relat"), ("triplicate", "triplic"), ("hopeful", "hope"),
    ("revival", "reviv"), ("adoption", "adopt"), ("probate", "probat"), ("controll", "control"),
])
def test_porter_paper_examples(word, expected):
    assert stem.porter_stem(word) == expected


def test_porter_agrees_with_nltk_original_mode():
    nltk_stem = pytest.importorskip("nltk.stem.porter")
    reference = nltk_stem.PorterStemmer(nltk_stem.PorterStemmer.ORIGINAL_ALGORITHM)
    import re, pathlib
    words = set(re.findall(r"\b[a-z]{3,}\b", pathlib.Path(np.__file__).parent.joinpath(
        "lib/_function_base_impl.py").read_text()))
    assert len(words) > 1000
    assert [w for w in words if stem.porter_stem(w) != reference.stem(w)] == []


def test_edit_distance_against_nltk():
    metrics = pytest.importorskip("nltk.metrics.distance")
    rng = random.Random(0)
    for _ in range(300):
        a = "".join(rng.choices("abcd", k=rng.randrange(8)))
        b = "".join(rng.choices("abcd", k=rng.randrange(8)))
        assert distance.levenshtein(a, b) == metrics.edit_distance(a, b)
        # Ours is optimal string alignment. NLTK's is unrestricted Damerau, which may
        # edit between transposed letters, so it can only be equal or smaller.
        osa = distance.damerau_levenshtein(a, b)
        assert metrics.edit_distance(a, b, transpositions=True) <= osa <= distance.levenshtein(a, b)


def test_alignment_cost_matches_distance():
    ops = distance.align("kitten", "sitting")
    assert sum(op != "=" for op, _, _ in ops) == 3


def test_spelling_prefers_frequent_word():
    corrector = distance.SpellingCorrector(["the"] * 100 + ["thaw"])
    assert corrector.correct("thw") == "the"


@pytest.mark.parametrize("sample", ["hello world", "naïve 東京 🚀", "  spaces\n\ttabs ", ""])
def test_bpe_round_trip(sample):
    tok = bpe.BPETokenizer().train(data.STORY, 320)
    assert tok.decode(tok.encode(sample)) == sample


def test_bpe_is_deterministic_and_compresses():
    a = bpe.BPETokenizer().train(data.STORY, 320)
    b = bpe.BPETokenizer().train(data.STORY, 320)
    assert a.merges == b.merges
    assert len(a.encode(data.STORY)) < len(data.STORY.encode()) / 1.8


SENTENCES = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat", "ran"]]


@pytest.mark.parametrize("kwargs", [
    {"n": 2, "smoothing": "laplace", "k": 0.5}, {"n": 3, "smoothing": "laplace", "k": 1.0},
    {"n": 2, "smoothing": "interpolation"}, {"n": 3, "smoothing": "interpolation"},
])
@pytest.mark.parametrize("context", [["the"], ["unseen"], ["the", "cat"], ["<s>", "<s>"]])
def test_smoothed_distributions_sum_to_one(kwargs, context):
    model = ngram.NGramModel(**kwargs).fit(SENTENCES)
    assert sum(model.prob(w, context) for w in model.vocab) == pytest.approx(1.0)


def test_mle_values_and_infinite_perplexity():
    model = ngram.NGramModel(2, "mle").fit(SENTENCES)
    assert model.prob("cat", ["the"]) == 0.5
    assert model.perplexity([["the", "ran"]]) == math.inf


def test_uniform_model_perplexity_is_vocabulary_size():
    model = ngram.NGramModel(1, "laplace", k=1e9).fit(SENTENCES)
    assert model.perplexity(SENTENCES) == pytest.approx(len(model.vocab), rel=1e-6)


def test_tfidf_matches_sklearn():
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    docs = [d.lower() for d in data.DOCUMENTS]
    ours = vectorspace.TfidfVectorizer(tokenizer=lambda d: d.split()).fit(docs)
    theirs = sk.TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False).fit(docs)
    assert sorted(ours.vocab) == sorted(theirs.vocabulary_)
    order = [ours.vocab[t] for t in theirs.get_feature_names_out()]
    assert np.allclose(ours.transform(docs)[:, order], theirs.transform(docs).toarray())


def test_bm25_hand_computed_score():
    docs = [["a", "b", "a"], ["b", "c"], ["c", "c", "d", "e"]]
    index = vectorspace.BM25(docs, k1=1.5, b=0.75)
    idf = math.log(1 + (3 - 1 + .5) / (1 + .5))
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - .75 + .75 * 3 / 3))
    assert dict(index.search(["a"]))[0] == pytest.approx(expected)


def test_average_precision():
    assert vectorspace.average_precision([1, 9, 2], {1, 2}) == pytest.approx((1 + 2 / 3) / 2)


def test_naive_bayes_matches_sklearn():
    sk = pytest.importorskip("sklearn.naive_bayes")
    docs = [d.split() for d, _ in data.REVIEWS]
    y = [l for _, l in data.REVIEWS]
    X = classify.BagOfWords().fit_transform(docs)
    ours, theirs = classify.MultinomialNB().fit(X, y), sk.MultinomialNB().fit(X, y)
    assert np.allclose(ours.feature_log_prob_, theirs.feature_log_prob_)
    assert np.allclose(ours.joint_log_likelihood(X), theirs.predict_joint_log_proba(X))


def test_bag_of_words_bigrams_match_sklearn():
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    docs = [d for d, _ in data.REVIEWS]
    theirs = sk.CountVectorizer(ngram_range=(1, 2), tokenizer=str.split, token_pattern=None).fit(docs)
    ours = classify.BagOfWords((1, 2)).fit([d.split() for d in docs])
    assert set(ours.vocab) == set(theirs.vocabulary_)


def test_logistic_regression_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    model = classify.LogisticRegression(epochs=200).fit(X, y.tolist())
    assert model.losses[-1] < model.losses[0] / 3
    assert np.mean(model.predict(X) == y) > 0.97


def test_group_splits_keep_groups_whole():
    docs, y, groups = list(range(100)), ["a"] * 50 + ["b"] * 50, data.REVIEW_GROUPS
    train, test, _, _ = classify.train_test_split(docs, y, 0.25, seed=0, groups=groups)
    assert not {groups[i] for i in train} & {groups[i] for i in test}


def test_metrics():
    report = classify.classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert report["a"]["precision"] == 1.0 and report["a"]["recall"] == 0.5
    assert report["accuracy"] == 0.75


TAGGER = hmm.HMMTagger().fit(data.pos_corpus(400, seed=0, rare=0.03))


@pytest.mark.parametrize("sentence", ["she will book .", "the train opened .", "a zorb ran today ."])
def test_viterbi_equals_brute_force(sentence):
    words = sentence.split()
    paths = list(itertools.product(TAGGER.tags, repeat=len(words)))
    best = max(TAGGER.joint_logprob(words, p) for p in paths)
    tags, score = TAGGER.viterbi(words)
    assert score == pytest.approx(best)
    assert TAGGER.joint_logprob(words, tags) == pytest.approx(best)


def test_forward_equals_explicit_sum():
    words = "she will book .".split()
    scores = [TAGGER.joint_logprob(words, p) for p in itertools.product(TAGGER.tags, repeat=len(words))]
    assert TAGGER.forward(words) == pytest.approx(float(np.logaddexp.reduce(scores)))


def test_context_disambiguates():
    assert TAGGER.viterbi("she will book the train .".split())[0][2] == "VERB"
    assert TAGGER.viterbi("the book opened .".split())[0][1] == "NOUN"


def test_transition_rows_are_distributions():
    totals = np.exp(TAGGER.log_trans).sum(axis=1) + np.exp(TAGGER.log_end)
    assert np.allclose(totals, 1.0)


def test_ppmi_is_zero_for_independent_events():
    independent = np.outer([1.0, 2.0], [3.0, 1.0])
    assert np.allclose(embeddings.ppmi(independent), 0.0)


def test_analogy_on_engineered_corpus():
    C, vocab = embeddings.cooccurrence_matrix(data.embedding_corpus(3000), 2)
    kv = embeddings.KeyedVectors(embeddings.svd_embeddings(embeddings.ppmi(C), 15), vocab)
    assert kv.analogy("man", "woman", "king")[0][0] == "queen"
