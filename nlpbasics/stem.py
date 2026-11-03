"""The Porter stemmer (Porter, 1980), written directly from the paper.

Stemming chops suffixes by rule so that connect, connected and connecting all
map to "connect". A stem need not be a word ("relational" -> "relat"). It only
has to be the SAME string for related words, which is what search needs.

Key concept: the measure m. Write any word as [C](VC)^m[V], where C is a run of
consonants and V a run of vowels. m approximates syllable count:

    tree -> m=0      trouble -> m=1      troubles -> m=2

Rules consult m on the remaining stem so that real roots do not get eaten:
"-ate" is removed from "activate" (m(activ)=2) but not from "plate" (m(pl)=0).

Compare lemmatization, which maps to dictionary forms (better -> good) and so
needs a vocabulary and usually a part of speech. Stemming needs neither.
"""
from __future__ import annotations

VOWELS = set("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    """'y' is a consonant at word start or after a vowel, otherwise a vowel:
    consonant in "toy" and "yes", vowel in "syzygy"."""
    c = word[i]
    if c in VOWELS:
        return False
    if c == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def measure(stem: str) -> int:
    """Count VC transitions: the m in [C](VC)^m[V]."""
    forms = [_is_consonant(stem, i) for i in range(len(stem))]
    return sum(1 for a, b in zip(forms, forms[1:]) if not a and b)


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _double_consonant(word: str) -> bool:
    return (len(word) >= 2 and word[-1] == word[-2]
            and _is_consonant(word, len(word) - 1))


def _cvc(word: str) -> bool:
    """Ends consonant-vowel-consonant with last letter not w, x or y (*o).
    This is the shape that needs an e restored: hop(e), fil(e)."""
    if len(word) < 3 or word[-1] in "wxy":
        return False
    n = len(word)
    return (_is_consonant(word, n - 3) and not _is_consonant(word, n - 2)
            and _is_consonant(word, n - 1))


def _step1a(w: str) -> str:
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _step1b(w: str) -> str:
    if w.endswith("eed"):
        return w[:-1] if measure(w[:-3]) > 0 else w
    for suffix in ("ed", "ing"):
        if w.endswith(suffix) and _has_vowel(w[:-len(suffix)]):
            w = w[:-len(suffix)]
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if _double_consonant(w) and w[-1] not in "lsz":
                return w[:-1]
            if measure(w) == 1 and _cvc(w):
                return w + "e"
            return w
        if w.endswith(suffix):
            return w
    return w


def _step1c(w: str) -> str:
    return w[:-1] + "i" if w.endswith("y") and _has_vowel(w[:-1]) else w


_STEP2 = {
    "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
    "izer": "ize", "abli": "able", "alli": "al", "entli": "ent", "eli": "e",
    "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate",
    "alism": "al", "iveness": "ive", "fulness": "ful", "ousness": "ous",
    "aliti": "al", "iviti": "ive", "biliti": "ble",
}
_STEP3 = {
    "icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic",
    "ful": "", "ness": "",
}
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize", "ion",
)


def _apply_longest(w: str, rules: dict[str, str]) -> str:
    """Only the LONGEST matching suffix is considered. If its condition fails,
    shorter suffixes do not get a second chance."""
    for suffix in sorted(rules, key=len, reverse=True):
        if w.endswith(suffix):
            stem = w[:-len(suffix)]
            return stem + rules[suffix] if measure(stem) > 0 else w
    return w


def _step4(w: str) -> str:
    for suffix in sorted(_STEP4, key=len, reverse=True):
        if w.endswith(suffix):
            stem = w[:-len(suffix)]
            if suffix == "ion" and not stem.endswith(("s", "t")):
                return w
            return stem if measure(stem) > 1 else w
    return w


def _step5(w: str) -> str:
    if w.endswith("e"):
        stem = w[:-1]
        m = measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    if measure(w) > 1 and _double_consonant(w) and w[-1] == "l":
        w = w[:-1]
    return w


def porter_stem(word: str) -> str:
    """Stem one lowercase word. Each step sees the output of the previous one,
    so "generalizations" -> "generalization" -> "generalize" -> "general"
    -> "gener". Words of one or two letters are returned as is."""
    w = word.lower()
    if len(w) <= 2:
        return w
    for step in (_step1a, _step1b, _step1c):
        w = step(w)
    w = _apply_longest(w, _STEP2)
    w = _apply_longest(w, _STEP3)
    return _step5(_step4(w))
