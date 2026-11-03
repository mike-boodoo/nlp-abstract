"""Lesson 1: from raw text to integers.
Read alongside: nlpbasics/text.py"""
from nlpbasics import data, text
from ._util import say, section, table, title


def main():
    title(1, "Text is messy: normalization and tokenization")

    section("Normalization: same appearance, different bytes")
    a, b = "café", "café"
    print(f"{a!r} == {b!r}  raw: {a == b}")
    print(f"after normalize: {text.normalize(a) == text.normalize(b)}")
    print(f"'ﬁnal' -> {text.normalize('ﬁnal')!r}   'STRASSE'/'straße' equal: "
          f"{text.normalize('STRASSE') == text.normalize('straße')}")
    print(f"strip_accents: {text.normalize('Crème Brûlée', strip_accents=True)!r}")

    section("Tokenization: why str.split() is not enough")
    sample = "Dr. Smith didn't pay $1,000.50 to the U.S. office; e-mail a.b@c.org (or see https://x.org/a?b=1)!"
    print("split():  ", sample.split())
    print("tokenize: ", text.tokenize(sample))
    print("contractions:", text.split_contractions(text.tokenize("She didn't say they'd go")))

    section("Sentence splitting: the ambiguous period")
    for s in text.split_sentences("Dr. Smith paid 3.5 dollars. He left at noon! Did she stay? Yes."):
        print("  ", repr(s))

    section("Stopwords: helpful, and dangerous")
    tokens = "this film is not good".split()
    print("kept:", text.remove_stopwords(tokens))
    say("'not' survives on purpose. Many stock stopword lists would delete it and "
        "flip the meaning of this review.")

    section("Zipf's law on a tiny corpus")
    tokens = text.tokenize(text.normalize(data.STORY))
    table(text.zipf_table(tokens, 8), ["rank", "token", "count", "rank*count"])
    vocab_counts = text.Vocabulary().fit([tokens]).counts
    once = sum(1 for c in vocab_counts.values() if c == 1)
    print(f"\n{len(tokens)} tokens, {len(vocab_counts)} types, {once} types seen exactly once "
          f"({once / len(vocab_counts):.0%}).")
    say("The rare tail is the central problem of statistical NLP: most word types "
        "have almost no evidence. Nearly every later lesson is a way of coping.")

    section("Vocabulary: tokens to integers")
    vocab = text.Vocabulary(min_freq=2).fit([tokens])
    ids = vocab.encode("the keeper saw a dragon".split())
    print("vocab size:", len(vocab))
    print("encode:", ids)
    print("decode:", vocab.decode(ids), " <- unseen words become <unk>; information is lost")


if __name__ == "__main__":
    main()
