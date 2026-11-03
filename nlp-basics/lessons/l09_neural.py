"""Lesson 9: backpropagation and a first neural language model.
Read alongside: nlpbasics/autograd.py, nlpbasics/neural.py"""
import numpy as np

from nlpbasics import data, ngram, neural, text
from nlpbasics.autograd import Tensor, cross_entropy
from ._util import say, section, table, title


def agreement_setup():
    """Shared with lesson 10."""
    train = data.agreement_corpus(4000, seed=0)
    test = data.agreement_corpus(300, seed=1, holdout=True)
    vocab = text.Vocabulary().fit(train)
    width = max(map(len, train + test))
    X = np.array([vocab.encode(s) + [0] * (width - len(s)) for s in train])
    return train, test, vocab, X, width


def has_distractor(sentence) -> bool:
    return sentence[3] == "saw"


def agreement_accuracy(model, vocab, sentences) -> float:
    """Is P(right pronoun) > P(wrong pronoun) given everything before it?"""
    his, her = vocab.stoi["his"], vocab.stoi["her"]
    correct = 0
    for s in sentences:
        i = next(k for k, w in enumerate(s) if w in ("his", "her"))
        p = neural.next_token_probs(model, vocab.encode(s[:i]))
        correct += (p[his] > p[her]) == (s[i] == "his")
    return correct / len(sentences)


def main():
    title(9, "Backpropagation and neural language models")

    section("Autograd: the chain rule, mechanized")
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)
    z = (x * y + y ** 2.0).tanh() * x
    z.backward()
    h = 1e-6
    f = lambda a, b: np.tanh(a * b + b ** 2) * a
    print(f"dz/dx  autograd {x.grad.item():.8f}  numerical {(f(3 + h, 2) - f(3 - h, 2)) / (2 * h):.8f}")
    print(f"dz/dy  autograd {y.grad.item():.8f}  numerical {(f(3, 2 + h) - f(3, 2 - h)) / (2 * h):.8f}")
    say("Backprop computes every partial derivative in one backward pass, at a cost "
        "comparable to the forward pass. Numerical differentiation needs a pass per parameter. "
        "It is too slow to train with, but it makes an excellent test.")

    section("Fitting softmax regression with autograd")
    rng = np.random.default_rng(0)
    data_x = rng.normal(size=(200, 2))
    labels = (data_x.sum(axis=1) > 0).astype(int)
    W = Tensor(np.zeros((2, 2)), requires_grad=True)
    for step in range(30):
        loss = cross_entropy(Tensor(data_x) @ W, labels)
        W.grad = None
        loss.backward()
        W.data -= 0.5 * W.grad
        if step % 10 == 0:
            print(f"  step {step:>2}  loss {loss.item():.3f}")

    section("A task that needs long-range context")
    train, test, vocab, X, width = agreement_setup()
    print("example:", " ".join(train[0]))
    print("example:", " ".join(train[1]))
    say("Predict the pronoun. The subject is seven or more words back, and half of "
        "the sentences have a distracting noun of the other gender nearer the pronoun. "
        "Test pairs of subject and clause never occur in training.")

    section("Baseline: trigram model")
    tri = ngram.NGramModel(3, "interpolation").fit(train)
    correct = 0
    for s in test:
        i = next(k for k, w in enumerate(s) if w in ("his", "her"))
        his, her = tri.prob("his", s[:i][-2:]), tri.prob("her", s[:i][-2:])
        correct += (his > her) == (s[i] == "his")
    print(f"pronoun accuracy: {correct / len(test):.3f}")

    section("Window MLP (Bengio et al., 2003)")
    rows = []
    for k in (3, 6, 12):
        model = neural.WindowMLP(len(vocab), context=k)
        losses = neural.train_language_model(model, X, steps=500)
        plain = [s for s in test if not has_distractor(s)]
        tricky = [s for s in test if has_distractor(s)]
        rows.append((k, model.num_parameters(), float(np.mean(losses[-50:])),
                     agreement_accuracy(model, vocab, test),
                     agreement_accuracy(model, vocab, plain),
                     agreement_accuracy(model, vocab, tricky)))
    table(rows, ["k", "params", "final loss", "accuracy", "no distractor", "distractor"])
    say("k=3 sees neither noun. k=12 sees the subject and solves the task. k=6 is the "
        "instructive row: it cannot reach the subject, yet it beats chance, because it sees "
        "the distractor, which in this dataset is always the opposite gender. It found a "
        "shortcut, a feature that predicts the label for an accidental reason. Accuracy alone "
        "would never have told you; slicing the test set by case did.")
    say("Big windows work, but parameters grow with k and k must be fixed in advance. Real "
        "dependencies have no such bound. Hence lesson 10.")


if __name__ == "__main__":
    main()
