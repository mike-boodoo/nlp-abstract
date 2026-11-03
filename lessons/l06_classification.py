"""Lesson 6: text classification, and how evaluation goes wrong.
Read alongside: nlpbasics/classify.py"""
import numpy as np

from nlpbasics import classify, data, text
from ._util import say, section, table, title


def analyze(doc: str) -> list[str]:
    return text.split_contractions(text.tokenize(text.normalize(doc)))


def main():
    title(6, "Naive Bayes, logistic regression, and evaluation")
    docs = [analyze(d) for d, _ in data.REVIEWS]
    labels = [l for _, l in data.REVIEWS]
    groups = data.REVIEW_GROUPS

    section("A puzzling result")
    nb = lambda: (classify.BagOfWords(), classify.MultinomialNB())
    naive = classify.cross_val_accuracy(nb, docs, labels)
    print(f"Naive Bayes, 5-fold CV, random folds: {np.mean(naive):.2f}")
    say("With two balanced classes a coin scores 0.50, so this is far WORSE than chance. "
        "Accuracy well below chance means there IS signal and we are reading it backwards.")
    print(f"\n  pos: {data.REVIEWS[1][0]}\n  neg: {data.REVIEWS[51][0]}")
    say("The dataset consists of near-twins with opposite labels. If one twin lands in "
        "training and the other in test, the many shared words vote for the training twin's "
        "label, which is the wrong one. Random splits ignore related items.")

    section("The fix: group-aware splits")
    grouped = classify.cross_val_accuracy(nb, docs, labels, groups=groups)
    print(f"Naive Bayes, twins kept together: {np.mean(grouped):.2f} "
          f"(folds: {[round(s, 2) for s in grouped]})")
    say("The everyday version of this error is the opposite and more dangerous: near-duplicates "
        "with the SAME label inflate your score, and nothing looks wrong. Always ask what "
        "connects your train and test items.")

    section("Comparing models with cross-validation")
    rows = []
    for name, factory in (
        ("NB, unigrams", lambda: (classify.BagOfWords(), classify.MultinomialNB())),
        ("NB, uni+bigrams", lambda: (classify.BagOfWords((1, 2)), classify.MultinomialNB())),
        ("LogReg, unigrams", lambda: (classify.BagOfWords(), classify.LogisticRegression())),
        ("LogReg, uni+bigrams", lambda: (classify.BagOfWords((1, 2)), classify.LogisticRegression())),
    ):
        scores = classify.cross_val_accuracy(factory, docs, labels, groups=groups)
        rows.append((name, float(np.mean(scores)), float(np.std(scores))))
    table(rows, ["model", "mean acc", "std"])
    say("The gaps are smaller than the fold-to-fold spread, so on 100 examples these models "
        "cannot be told apart. The honest conclusion is 'no detectable difference', not a winner.")

    section("Inside the models")
    Xtr_docs, Xte_docs, ytr, yte = classify.train_test_split(docs, labels, 0.25, seed=1, groups=groups)
    vec = classify.BagOfWords()
    Xtr, Xte = vec.fit_transform(Xtr_docs), vec.transform(Xte_docs)
    names = vec.feature_names
    model = classify.MultinomialNB().fit(Xtr, ytr)
    for label, top in model.most_indicative(names, 8).items():
        print(f"NB most {label}: {[w for w, _ in top]}")
    logreg = classify.LogisticRegression().fit(Xtr, ytr)
    print(f"LogReg loss: {logreg.losses[0]:.3f} -> {logreg.losses[-1]:.3f} "
          f"over {len(logreg.losses)} epochs")
    print(f"train accuracy {np.mean(logreg.predict(Xtr) == np.array(ytr)):.2f}, "
          f"test accuracy {np.mean(logreg.predict(Xte) == np.array(yte)):.2f}")
    say("That gap is overfitting: far more features than examples.")

    section("Precision, recall, F1")
    predicted = model.predict(Xte).tolist()
    labels_, matrix = classify.confusion_matrix(yte, predicted)
    table([[labels_[i], *matrix[i].tolist()] for i in range(len(labels_))],
          ["true \\ pred", *labels_])
    report = classify.classification_report(yte, predicted)
    print()
    table([(c, report[c]["precision"], report[c]["recall"], report[c]["f1"], report[c]["support"])
           for c in labels_], ["class", "precision", "recall", "f1", "n"])

    section("Why accuracy can lie")
    truth, always_ham = ["ham"] * 99 + ["spam"], ["ham"] * 100
    r = classify.classification_report(truth, always_ham)
    print(f"always predict ham on 99% ham: accuracy {r['accuracy']:.2f}, "
          f"spam recall {r['spam']['recall']:.2f}, macro F1 {r['macro_f1']:.2f}")

    section("Bag-of-words sees no order")
    tests = ["not good", "good", "not bad"]
    vectors = classify.BagOfWords().fit(Xtr_docs)
    for sentence in tests:
        p = logreg.predict_proba(vectors.transform([sentence.split()]))[0]
        print(f"P(pos | {sentence!r:11}) = {p[list(logreg.classes_).index('pos')]:.2f}")
    say("'not' is one feature with one weight, pushing every phrase the same way. It cannot "
        "flip 'bad' into a compliment. Bigrams would need 'not bad' in training, which is "
        "why sequence-aware models matter for sentiment.")


if __name__ == "__main__":
    main()
