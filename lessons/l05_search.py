"""Lesson 5: documents as vectors, and a working search engine.
Read alongside: nlpbasics/vectorspace.py"""
import numpy as np

from nlpbasics import data, stem, text, vectorspace
from ._util import say, section, table, title


def analyze(doc: str, stemmed: bool = True) -> list[str]:
    tokens = text.remove_stopwords(t for t in text.tokenize(text.normalize(doc)) if t.isalnum())
    return [stem.porter_stem(t) for t in tokens] if stemmed else tokens


# Relevance judgments, assigned by hand. Doc ids index data.DOCUMENTS.
QUERIES = {
    "how do plants make sugar": {0},
    "interest rates and inflation": {5, 6},
    "movement in the crust of the earth": {2, 3},
    "training models with gradients": {7, 8},
    "books spreading across europe": {11},
    "yeast dough rising": {12},
    "animals that live in the sea and breathe": {15},
}


def main():
    title(5, "TF-IDF, BM25 and ranked retrieval")
    docs = data.DOCUMENTS

    section("IDF: rarity as information")
    vec = vectorspace.TfidfVectorizer(tokenizer=lambda d: analyze(d, False)).fit(docs)
    words = sorted(vec.vocab, key=lambda t: vec.idf[vec.vocab[t]])
    print("lowest idf: ", [(w, round(float(vec.idf[vec.vocab[w]]), 2)) for w in words[:3]])
    print("highest idf:", [(w, round(float(vec.idf[vec.vocab[w]]), 2)) for w in words[-3:]])

    section("Cosine similarity between documents")
    X = vec.transform(docs)
    sims = X @ X.T
    np.fill_diagonal(sims, 0)
    for i, j in sorted({tuple(sorted((a, int(sims[a].argmax())))) for a in (2, 5, 7, 12)}):
        print(f"{sims[i, j]:.2f}  [{i}] {docs[i][:44]}...  <->  [{j}] {docs[j][:44]}...")

    section("A query against BM25")
    index = vectorspace.BM25([analyze(d) for d in docs])
    query = "why do interest rates rise"
    print("query:", query, "->", analyze(query))
    for doc_id, score in index.search(analyze(query), 3):
        print(f"  {score:5.2f}  [{doc_id}] {docs[doc_id]}")

    section("BM25 saturation")
    table([(tf, tf * 2.5 / (tf + 1.5)) for tf in (1, 2, 3, 5, 10, 50)], ["tf", "tf factor at average length"])
    say("Term frequency has diminishing returns, approaching k1 + 1. Under raw TF-IDF a "
        "keyword-stuffed document just keeps gaining.")

    section("Evaluation: mean average precision over judged queries")
    def rank_tfidf(stemmed):
        v = vectorspace.TfidfVectorizer(tokenizer=lambda d: analyze(d, stemmed))
        matrix = v.fit_transform(docs)
        return lambda q: list(np.argsort(-(matrix @ v.transform([q])[0])))

    def rank_bm25(stemmed):
        bm25 = vectorspace.BM25([analyze(d, stemmed) for d in docs])
        return lambda q: [d for d, _ in bm25.search(analyze(q, stemmed), len(docs))]

    rows = []
    for name, ranker in (("TF-IDF, no stemming", rank_tfidf(False)), ("TF-IDF, stemmed", rank_tfidf(True)),
                         ("BM25, no stemming", rank_bm25(False)), ("BM25, stemmed", rank_bm25(True))):
        ap = [vectorspace.average_precision(ranker(q), rel) for q, rel in QUERIES.items()]
        rows.append((name, float(np.mean(ap)), sum(a == 1.0 for a in ap)))
    table(rows, ["system", "MAP", f"perfect / {len(QUERIES)}"])
    say("Seven queries over sixteen documents show the method, not a meaningful ranking of "
        "systems. Published comparisons use thousands of judged queries.")

    section("Limits: bag of words has no idea of meaning")
    print("'dog bites man' vs 'man bites dog' cosine:",
          round(vectorspace.cosine(*vectorspace.TfidfVectorizer().fit_transform(
              ["dog bites man", "man bites dog"])), 3))
    results = index.search(analyze("canine"), 3)
    print("query 'canine' results:", results, " <- synonyms never match")
    say("Identical bags, opposite meanings; related words, zero overlap. "
        "Lesson 8 addresses the second problem and lesson 10 the first.")


if __name__ == "__main__":
    main()
