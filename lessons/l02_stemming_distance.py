"""Lesson 2: word forms and word similarity.
Read alongside: nlpbasics/stem.py, nlpbasics/distance.py"""
from nlpbasics import data, distance, stem, text
from ._util import say, section, table, title


def main():
    title(2, "Stemming, edit distance, spelling correction")

    section("The Porter measure m in [C](VC)^m[V]")
    table([(w, stem.measure(w)) for w in ("tree", "trouble", "oats", "troubles", "private")],
          ["word", "m"])

    section("Porter stemming, step by step")
    table([(w, stem.porter_stem(w)) for w in
           ("caresses", "ponies", "connected", "connecting", "connection", "relational",
            "generalizations", "hopping", "sky", "university", "universe", "organ", "organization")],
          ["word", "stem"])
    say("Stems need not be words: 'ponies' -> 'poni', and 'pony' -> "
        f"'{stem.porter_stem('pony')}' too, which is the point. Overstemming merges unrelated "
        "words: 'university' with 'universe', 'organ' with 'organization'. Understemming "
        "fails to merge related ones:")
    print("  alumnus ->", stem.porter_stem("alumnus"), "  alumni ->", stem.porter_stem("alumni"))

    section("Edit distance as a dynamic program")
    a, b = "kitten", "sitting"
    D = distance.edit_matrix(a, b)
    table([[(" " + a)[i]] + D[i] for i in range(len(a) + 1)], ["", "_", *b])
    print(f"\ndistance = {D[-1][-1]}")
    print("edit script:", [op for op in distance.align(a, b) if op[0] != "="])

    section("Transpositions")
    print("levenshtein('teh', 'the')        =", distance.levenshtein("teh", "the"))
    print("damerau_levenshtein('teh', 'the') =", distance.damerau_levenshtein("teh", "the"))

    section("Word error rate: the same algorithm on word lists")
    reference = "the cat sat on the mat".split()
    hypothesis = "the cat sat on mat today".split()
    errors = distance.levenshtein(reference, hypothesis)
    print(f"WER = {errors}/{len(reference)} = {errors / len(reference):.2f}")

    section("Noisy channel spelling correction")
    corrector = distance.SpellingCorrector(text.tokenize(text.normalize(data.STORY)))
    table([(w, corrector.correct(w)) for w in ("keper", "lighthose", "thw", "stoem", "wether", "qqqq")],
          ["typed", "corrected"])
    scored = corrector.candidates("hed")
    print("\nscores for 'hed':", {w: round(s, 2) for w, s in sorted(scored.items(), key=lambda x: -x[1])[:4]})
    say("Every candidate is one edit away, so the error model cannot decide. The language "
        "model (word frequency) does. Unfixable input is returned unchanged.")


if __name__ == "__main__":
    main()
