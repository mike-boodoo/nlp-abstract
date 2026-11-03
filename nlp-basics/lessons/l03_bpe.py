"""Lesson 3: subword tokenization, as used by modern language models.
Read alongside: nlpbasics/bpe.py"""
from nlpbasics import bpe, data
from ._util import say, section, table, title


def main():
    title(3, "Byte Pair Encoding")
    corpus = " ".join(data.STORY.split())

    section("Training: watch merges build words")
    tok = bpe.BPETokenizer().train(corpus, 400)
    first = [(i + 1, repr(tok.vocab[new_id].decode("utf-8", "replace")))
             for i, new_id in enumerate(list(tok.merges.values())[:12])]
    table(first, ["merge #", "new token"])
    print("later merges:", [tok.vocab[m].decode("utf-8", "replace")
                            for m in list(tok.merges.values())[-6:]])

    section("Encoding")
    sample = "The keeper lit the lamp."
    print(tok.tokens(sample))

    section("Never out of vocabulary")
    for s in ("The lighthousekeeper's lamp", "naïve 東京 🚀"):
        ids = tok.encode(s)
        print(f"{s!r}: {len(ids)} ids; lossless round trip: {tok.decode(ids) == s}")
    say("Unknown material falls back to smaller pieces, ultimately raw bytes, "
        "so there is no <unk> token. The cost is more tokens, not lost information.")

    section("Vocabulary size is a compression trade-off")
    raw = len(corpus.encode("utf-8"))
    rows = []
    for size in (256, 300, 400, 600, 1000):
        t = bpe.BPETokenizer().train(corpus, size)
        n = len(t.encode(corpus))
        rows.append((size, n, raw / n))
    table(rows, ["vocab", "tokens", "bytes/token"])
    say("Larger vocabularies give shorter sequences, until no pair repeats and training "
        "stops early: this corpus runs out of merges before 1000. Production tokenizers "
        "train tens of thousands of merges on far more text.")

    section("A consequence you can see")
    print("'keeper' ->", tok.tokens("keeper"), "   ' keeper' ->", tok.tokens(" keeper"))
    print("'KEEPER' ->", tok.tokens("KEEPER"))
    say("Tokens are artifacts of training frequency, not linguistic units. The same word "
        "tokenizes differently by case and spacing. It is one reason language models can "
        "be weak at spelling and letter-counting tasks: they often never see the letters.")


if __name__ == "__main__":
    main()
