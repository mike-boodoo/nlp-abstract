"""Lesson 4: probability over sequences.
Read alongside: nlpbasics/ngram.py"""
import math

from nlpbasics import data, ngram, text
from ._util import say, section, table, title


def main():
    title(4, "N-gram language models")
    sentences = [text.tokenize(text.normalize(s)) for s in text.split_sentences(" ".join(data.STORY.split()))]
    train, dev, test = sentences[:-8], sentences[-8:-4], sentences[-4:]
    print(f"{len(train)} training, {len(dev)} dev, {len(test)} test sentences")

    section("Maximum likelihood, by hand")
    bigram = ngram.NGramModel(2, "mle").fit(train)
    for word, ctx in (("keeper", "old"), ("lamp", "the"), ("keeper", "the")):
        print(f"P({word} | {ctx}) = {bigram.prob(word, [ctx]):.3f}")

    section("The zero problem")
    unseen = "the cat lit the fish .".split()
    print(f"sentence {unseen}")
    print(f"MLE log P = {bigram.logprob(unseen)}")
    say("One unseen bigram makes the entire sentence impossible, and the test perplexity "
        "infinite.")

    section("Smoothing compared on held-out text (lower is better)")
    rows = []
    specs = [("unigram, MLE", 1, "mle", {}), ("bigram, MLE", 2, "mle", {}),
             ("bigram, add-1", 2, "laplace", {"k": 1.0}), ("bigram, add-0.01", 2, "laplace", {"k": 0.01}),
             ("bigram, interpolated", 2, "interpolation", {}),
             ("trigram, add-0.01", 3, "laplace", {"k": 0.01}),
             ("trigram, interpolated", 3, "interpolation", {})]
    for name, n, smoothing, kwargs in specs:
        model = ngram.NGramModel(n, smoothing, min_count=2, **kwargs).fit(train)
        rows.append((name, model.perplexity(train), model.perplexity(test)))
    table(rows, ["model", "train perplexity", "test perplexity"])
    say("Train scores always favor more context. Only the held-out column is honest. "
        "This is the bias-variance trade-off: longer contexts are more specific, but they "
        "have fewer examples. Note how badly add-k trigrams do on test, where most contexts "
        "are unseen, while interpolation falls back to shorter contexts and stays stable.")

    section("Tuning on dev, reporting on test")
    best, dev_score = ngram.grid_search_lambdas(train, dev)
    tuned = ngram.NGramModel(3, "interpolation", lambdas=best, min_count=2).fit(train)
    print(f"best lambdas (tri, bi, uni): {[round(x, 2) for x in best]}  dev perplexity {dev_score:.2f}")
    print(f"test perplexity {tuned.perplexity(test):.2f}")

    section("Generation: more context, more plausible")
    for n in (1, 2, 3):
        model = ngram.NGramModel(n, "laplace", k=0.01).fit(train)
        print(f"n={n}:", " ".join(model.generate(max_len=16, seed=3)))
    say("Higher orders look more like English because, on a corpus this small, they are "
        "splicing together chunks of training text. They recombine; they do not understand.")

    section("Why we add logs")
    print(f"0.1 ** 400 as a product: {0.1 ** 400}   as a log: {400 * math.log(0.1):.1f}")


if __name__ == "__main__":
    main()
