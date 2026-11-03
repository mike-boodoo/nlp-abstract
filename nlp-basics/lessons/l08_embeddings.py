"""Lesson 8: meaning as geometry.
Read alongside: nlpbasics/embeddings.py"""
import numpy as np

from nlpbasics import data, embeddings
from ._util import say, section, table, title


def main():
    title(8, "Word embeddings: PPMI + SVD and word2vec")
    corpus = data.embedding_corpus()
    print(f"{len(corpus)} sentences, e.g. {' '.join(corpus[0])!r}")
    say("This corpus is engineered with clean gender, royalty and age axes. Results here show "
        "the mechanism working under ideal conditions, not performance on natural text.")

    section("One-hot vectors know nothing")
    C, vocab = embeddings.cooccurrence_matrix(corpus, window=2)
    i = {w: k for k, w in enumerate(vocab)}
    eye = np.eye(len(vocab))
    print(f"cos(cat, dog)    one-hot: {eye[i['cat']] @ eye[i['dog']]:.1f}")
    print(f"cos(cat, bread)  one-hot: {eye[i['cat']] @ eye[i['bread']]:.1f}")

    section("Raw counts overrate frequent words; PPMI corrects")
    P = embeddings.ppmi(C)
    def strongest(M, word):
        return [vocab[k] for k in np.argsort(-M[i[word]])[:5]]
    print("'crown' by raw count:", strongest(C, "crown"))
    print("'crown' by PPMI:     ", strongest(P, "crown"))

    count_vectors = embeddings.KeyedVectors(embeddings.svd_embeddings(P, 15), vocab)
    section("Prediction-based vectors: skip-gram with negative sampling")
    model = embeddings.SkipGram(dim=30, epochs=4).fit(corpus)
    print("loss by epoch:", [round(x, 3) for x in model.losses])
    sgns = embeddings.KeyedVectors(model.W, model.vocab)

    section("Nearest neighbors")
    rows = [(w, ", ".join(n for n, _ in count_vectors.nearest(w, 3)),
             ", ".join(n for n, _ in sgns.nearest(w, 3))) for w in ("cat", "bread", "paris", "king")]
    table(rows, ["word", "PPMI+SVD", "skip-gram"])

    section("Analogies: a is to b as c is to ?")
    queries = [("man", "woman", "king", "queen"), ("king", "queen", "prince", "princess"),
               ("man", "king", "woman", "queen"), ("boy", "girl", "man", "woman")]
    rows = [(f"{a}:{b} :: {c}:?", want, count_vectors.analogy(a, b, c)[0][0], sgns.analogy(a, b, c)[0][0])
            for a, b, c, want in queries]
    table(rows, ["query", "expected", "PPMI+SVD", "skip-gram"])

    section("Words have one vector, but several senses")
    say("Every embedding here assigns one vector per word type. A word like 'bank' would "
        "average its river and money senses into a single point. Contextual models address "
        "this: attention, next, computes a word's representation from its sentence.")

    section("Embeddings learn the corpus, including its biases")
    print(f"cos(king, he)  {sgns.similarity('king', 'he'):+.2f}")
    print(f"cos(king, she) {sgns.similarity('king', 'she'):+.2f}")
    say("Here that association was typed into the generator. In models trained on natural "
        "text the same mechanism absorbs real social stereotypes, as Bolukbasi et al. (2016) "
        "documented for occupations in word2vec.")


if __name__ == "__main__":
    main()
