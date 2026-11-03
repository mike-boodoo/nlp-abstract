"""Text preprocessing: the unglamorous layer every NLP system stands on.

Core idea: a computer sees text as code points, not words. Every later model
inherits the decisions made here. If "Cafe" and "café" become different tokens,
nothing downstream can know they are the same word.

Pipeline covered:  raw text -> normalize -> sentences -> tokens -> integer ids
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence


def normalize(text: str, lowercase: bool = True, strip_accents: bool = False) -> str:
    """Canonicalize text so visually identical strings compare equal.

    NFKC folds compatibility forms: the single ligature character "ﬁ" becomes
    "fi" and a full-width "Ａ" becomes "A". casefold() is a stronger lower():
    German "ß" becomes "ss", which lower() leaves alone.

    strip_accents works by decomposing to NFD (é -> e + U+0301) and dropping the
    combining marks. It improves recall and destroys information: in Spanish
    "año" (year) and "ano" (anus) become one token. Choose per application.
    """
    text = unicodedata.normalize("NFKC", text)
    if strip_accents:
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


# Order matters: regex alternation takes the FIRST branch that matches, not the
# longest, so specific patterns (URLs) must precede general ones (words).
TOKEN_RE = re.compile(
    r"""
    https?://\S*[^\s.,;:!?)\]'"]     # URLs, minus trailing punctuation
  | [\w.+-]+@[\w-]+\.[\w.-]*\w       # e-mail addresses
  | (?:[A-Za-z]\.){2,}               # abbreviations: U.S., e.g.
  | \$?\d+(?:[.,]\d+)*%?             # numbers: 3.14, 1,000, $5, 20%
  | \w+(?:[-'’]\w+)*                 # words, with inner hyphens or apostrophes
  | \.\.\.                           # ellipsis
  | [^\w\s]                          # any other single symbol
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[str]:
    """Regex tokenizer. Compare with the naive baseline, text.split(), which
    glues punctuation onto words so that "dog" and "dog." differ."""
    return TOKEN_RE.findall(text)


_CONTRACTION_RE = re.compile(r"(?i)^(.+?)(n['’]t|['’](?:s|re|ve|ll|d|m))$")


def split_contractions(tokens: Iterable[str]) -> list[str]:
    """Penn Treebank style: don't -> do + n't, she's -> she + 's.

    Useful because "n't" is a separate unit of meaning (negation) that a
    sentiment model can then learn once, rather than once per verb.
    """
    out: list[str] = []
    for tok in tokens:
        match = _CONTRACTION_RE.match(tok)
        out.extend(match.groups() if match else [tok])
    return out


ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd",
    "e.g", "i.e", "u.s", "fig", "no",
}


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter.

    A period is ambiguous: it can end a sentence, an abbreviation, or be part of
    a number. Rule: split after [.!?] when followed by whitespace and an
    uppercase letter, digit or opening quote, unless the word before the period
    is a known abbreviation. Real systems (Punkt, spaCy) learn these decisions,
    but they are answering this same question.
    """
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text):
        chunk = text[start:match.start() + 1]
        last_word = chunk.split()[-1].rstrip(".!?").lower() if chunk.split() else ""
        if text[match.start()] == "." and last_word in ABBREVIATIONS:
            continue
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have he her his i if in into
    is it its me my of on or our she so than that the their them then there these
    they this to us was we were what when which who will with you your""".split()
)


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Drop high-frequency function words.

    Helps keyword search; can actively hurt elsewhere. Note that "not" is
    deliberately absent from STOPWORDS: many stock lists include it, and
    removing it turns "not good" into "good".
    """
    return [t for t in tokens if t.lower() not in STOPWORDS]


class Vocabulary:
    """Bidirectional token <-> integer map. Models consume numbers, not strings.

    Special tokens:
      <pad>  fills short sequences so batches are rectangular (id 0)
      <unk>  stands in for anything unseen or too rare (id 1)

    min_freq sends rare tokens to <unk>. That has a second benefit beyond
    a smaller table: the training data now contains <unk>, so models learn
    what to do with an unknown word before they meet one at test time.
    """

    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, min_freq: int = 1, max_size: int | None = None,
                 specials: Sequence[str] = ()):
        self.min_freq, self.max_size = min_freq, max_size
        self.itos: list[str] = [self.PAD, self.UNK, *specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.counts: Counter[str] = Counter()

    def fit(self, documents: Iterable[Sequence[str]]) -> "Vocabulary":
        for doc in documents:
            self.counts.update(doc)
        # Sort by (-count, token) so the mapping is deterministic between runs.
        ranked = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, count in ranked:
            if count < self.min_freq or (self.max_size and len(self) >= self.max_size):
                break
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        return self

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def zipf_table(tokens: Iterable[str], top: int = 10) -> list[tuple[int, str, int, int]]:
    """Return (rank, token, count, rank*count).

    Zipf's law: frequency is roughly proportional to 1/rank, so rank*count stays
    within the same order of magnitude. It is why a few words dominate, why most
    vocabulary types are rare, and thus why every statistical model needs a plan
    for words it has seen rarely or never.
    """
    ranked = Counter(tokens).most_common(top)
    return [(r, tok, n, r * n) for r, (tok, n) in enumerate(ranked, 1)]
