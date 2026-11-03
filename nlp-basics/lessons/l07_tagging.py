"""Lesson 7: labeling sequences with hidden Markov models.
Read alongside: nlpbasics/hmm.py"""
import itertools

import numpy as np

from nlpbasics import data, hmm
from ._util import say, section, table, title


def main():
    title(7, "Part-of-speech tagging with HMMs and Viterbi")
    train = data.pos_corpus(2000, seed=0, rare=0.03)
    plain = data.pos_corpus(2000, seed=0)
    test = data.pos_corpus(400, seed=1, oov=0.2)
    known = {w for s in train for w, _ in s}
    tagger = hmm.HMMTagger().fit(train)
    print("tags:", tagger.tags)
    print("sample:", train[0])

    section("Context resolves ambiguity")
    for sentence in ("she will book the train .", "the book near a train opened .",
                     "they watched the light .", "he will light the small light ."):
        words = sentence.split()
        tags, _ = tagger.viterbi(words)
        print("  " + " ".join(f"{w}/{t}" for w, t in zip(words, tags)))

    section("Why dynamic programming: the search space")
    T = len(tagger.tags)
    table([(n, f"{T ** n:.2e}", n * T * T) for n in (5, 10, 20, 40)], ["words", "T^n paths", "n*T^2 steps"])

    section("Viterbi agrees with exhaustive search")
    words = "she will book the train .".split()
    best = max(itertools.product(tagger.tags, repeat=len(words)),
               key=lambda tags: tagger.joint_logprob(words, tags))
    print("brute force:", list(best))
    print("viterbi:    ", tagger.viterbi(words)[0])

    section("Forward algorithm: sum instead of max")
    total = np.logaddexp.reduce([tagger.joint_logprob(words, tags)
                                 for tags in itertools.product(tagger.tags, repeat=len(words))])
    print(f"forward  log P(words) = {tagger.forward(words):.4f}")
    print(f"explicit sum of {T ** len(words):,} paths = {total:.4f}")

    section("Evaluation on held-out text with unseen words")
    baseline = hmm.MostFrequentTagBaseline().fit(train)
    systems = [("most-frequent-tag baseline", baseline.tag, train),
               ("HMM, no rare words in training", hmm.HMMTagger().fit(plain), plain),
               ("HMM, rare words -> one <UNK>", hmm.HMMTagger(use_shapes=False).fit(train), train),
               ("HMM, rare words -> shape classes", tagger, train)]
    rows = []
    for name, model, corpus in systems:
        vocabulary = {w for s in corpus for w, _ in s}
        predict = model if callable(model) else (lambda w, m=model: m.viterbi(w)[0])
        result = hmm.tagging_accuracy(predict, test, vocabulary)
        rows.append((name, result["accuracy"], result["oov_accuracy"]))
    table(rows, ["system", "accuracy", "unseen-word accuracy"])
    say("Sequence context beats the baseline. Training on rare words teaches the model how "
        "unknown words behave, which accounts for the largest gain on unseen words. Shape classes "
        "add nothing measurable here because this grammar's context "
        "already determines every tag. The classes do hold real information, though, as "
        "the tag most likely to emit each one shows:")
    for shape in ("<UNK-ed>", "<UNK-ing>", "<UNK-ly>", "<UNK-NUM>"):
        column = tagger.log_emit[:, tagger.word_index[shape]]
        print(f"  {shape:<10} -> {tagger.tags[int(column.argmax())]}")
    say("Natural text is far less tidy than this grammar, so such evidence matters more there. "
        "Near-perfect accuracy reflects a simple synthetic grammar, not an expectation for "
        "natural text.")


if __name__ == "__main__":
    main()
