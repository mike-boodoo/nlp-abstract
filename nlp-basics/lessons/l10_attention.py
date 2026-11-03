"""Lesson 10: recurrence, attention, and a tiny transformer.
Read alongside: nlpbasics/neural.py"""
import time

import numpy as np

from nlpbasics import neural
from nlpbasics.autograd import Tensor
from ._util import say, section, table, title
from .l09_neural import agreement_accuracy, agreement_setup


def main():
    title(10, "From RNNs to attention to transformers")
    train, test, vocab, X, width = agreement_setup()

    section("Attention by hand")
    q = Tensor(np.array([[[1.0, 0.0]]]))                    # what I am looking for
    k = Tensor(np.array([[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]]))  # what each item offers
    v = Tensor(np.array([[[10.0], [20.0], [30.0]]]))        # what each item contains
    out, weights = neural.scaled_dot_product_attention(q, k, v)
    print("weights:", np.round(weights[0, 0], 3))
    print("output: ", np.round(out.data[0, 0], 3), "<- a weighted average of the values")

    section("The causal mask")
    print(np.triu(np.ones((5, 5), dtype=int), k=1))
    say("Row t may not attend where this shows 1, so a position cannot read the future "
        "token it must predict.")

    section("Same task, three architectures")
    rows, models = [], {}
    for name, model in (("Elman RNN", neural.RNNLM(len(vocab), cell="elman")),
                        ("GRU", neural.RNNLM(len(vocab), cell="gru")),
                        ("Transformer", neural.TinyTransformer(len(vocab), max_len=width))):
        start = time.time()
        losses = neural.train_language_model(model, X, steps=800)
        rows.append((name, model.num_parameters(), float(np.mean(losses[-50:])),
                     agreement_accuracy(model, vocab, test), f"{time.time() - start:.0f}s"))
        models[name] = model
    table(rows, ["model", "params", "final loss", "pronoun accuracy", "time"])
    say("Unlike the window models, none of these has a parameter count tied to how far back "
        "the subject sits. This gap is short enough that even the plain RNN copes. To see "
        "recurrent models fail as the gap grows, run experiments/gap_length.py (a few minutes).")

    section("Where the transformer looks")
    model = models["Transformer"]
    for subject in ("queen", "king"):
        words = f"the {subject} who saw the boy by the river lost".split()
        ids = vocab.encode(words)
        p = neural.next_token_probs(model, ids)
        maps = model.attention_maps(ids)
        print(f"\n{' '.join(words)} ___")
        print(f"P(her)={p[vocab.stoi['her']]:.3f}  P(his)={p[vocab.stoi['his']]:.3f}")
        for layer, weights in enumerate(maps, 1):
            row = weights[-1]
            top = ", ".join(f"{words[i]}@{i}={row[i]:.2f}" for i in np.argsort(-row)[:3])
            print(f"  layer {layer} top attention from final word: {top}"
                  f"  (subject@1={row[1]:.2f})")
    say("Only the subject differs between these sentences, and the prediction flips, so the "
        "model does use it. But do not expect the weights to point neatly at the subject: "
        "information can hop through intermediate positions across layers. Attention maps "
        "are a hint about a model, not an explanation of it. Changing an input and watching "
        "the output, as above, is stronger evidence.")

    section("Sampling: temperature")
    prompt = vocab.encode("the princess who".split())
    for temperature in (0.3, 1.0, 2.0):
        ids = neural.sample(model, prompt, max_new=12, temperature=temperature,
                            stop_id=vocab.stoi["."], seed=1, max_len=model.max_len)
        print(f"T={temperature}: {' '.join(vocab.decode(ids))}")

    section("Scale")
    print(f"this model: {model.num_parameters():,} parameters, {len(vocab)} word vocabulary")
    say("A GPT-style language model has the same parts: BPE tokens (lesson 3), embeddings "
        "(lesson 8), cross-entropy trained by backprop (lesson 9), and causal self-attention "
        "(here). The large differences are in scale, data and engineering. "
        "You have now built each piece.")


if __name__ == "__main__":
    main()
