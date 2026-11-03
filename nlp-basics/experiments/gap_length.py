"""How far can each architecture carry information?

Stretches the agreement task with extra clauses and retrains. Takes a few minutes.
Run:  python -m experiments.gap_length

Single seed and a fixed training budget, so treat the numbers as an illustration.
The architectures also differ in size and cost per step.
"""
import numpy as np

from lessons._util import table
from lessons.l09_neural import agreement_accuracy
from nlpbasics import data, neural, text


def main() -> None:
    rows = []
    for extra in (0, 3, 6):
        train = data.agreement_corpus(3000, 0, extra_clauses=extra)
        test = data.agreement_corpus(150, 1, holdout=True, extra_clauses=extra)
        vocab = text.Vocabulary().fit(train)
        width = max(map(len, train + test))
        X = np.array([vocab.encode(s) + [0] * (width - len(s)) for s in train])
        row = [extra, width]
        for model in (neural.RNNLM(len(vocab), cell="elman"), neural.RNNLM(len(vocab), cell="gru"),
                      neural.TinyTransformer(len(vocab), max_len=width)):
            neural.train_language_model(model, X, steps=800, batch_size=16)
            row.append(agreement_accuracy(model, vocab, test))
        rows.append(row)
        print(f"finished extra_clauses={extra}", flush=True)
    table(rows, ["extra clauses", "max tokens", "Elman", "GRU", "Transformer"])


if __name__ == "__main__":
    main()
