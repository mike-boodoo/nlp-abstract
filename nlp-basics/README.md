# NLP Basics, From Scratch

A runnable course on the foundations of natural language processing. Every algorithm is
implemented from first principles in readable Python, from regex tokenizers to a working
transformer with its own autograd engine. Only NumPy is required, and no model downloads
or internet access are needed.

```bash
pip install -r requirements.txt
python run_all.py            # all 10 lessons, about 90 seconds
python run_all.py 4 7        # selected lessons
python -m exercises.check    # your exercise progress
```

## How to study this

1. Read a lesson's module in `nlpbasics/`. The docstrings are the textbook.
2. Run the lesson with `python run_all.py N` and predict each output before reading it.
3. Solve the matching function in `exercises/problems.py`.
4. Change something (smoothing, window size, a dataset) and find out what breaks.

## Curriculum

| # | Topic | You build | Code | Key idea |
|---|-------|-----------|------|----------|
| 1 | Text processing | Normalizer, tokenizer, sentence splitter, vocabulary | `text.py` | Every model inherits your preprocessing choices |
| 2 | Word forms | Porter stemmer, edit distance, spelling corrector | `stem.py`, `distance.py` | Dynamic programming; the noisy channel |
| 3 | Subword tokens | Byte Pair Encoding | `bpe.py` | How modern language models see text |
| 4 | Language models | N-grams, smoothing, perplexity | `ngram.py` | Unseen events; evaluate on held-out data |
| 5 | Retrieval | TF-IDF, BM25, inverted index, MAP | `vectorspace.py` | Documents as vectors |
| 6 | Classification | Naive Bayes, logistic regression, metrics, CV | `classify.py` | How evaluation goes wrong |
| 7 | Sequence labeling | HMM, Viterbi, forward algorithm | `hmm.py` | Joint decisions at polynomial cost |
| 8 | Word embeddings | PPMI + SVD, skip-gram with negative sampling | `embeddings.py` | Meaning from co-occurrence |
| 9 | Neural basics | Reverse-mode autograd, Adam, window MLP | `autograd.py`, `neural.py` | Backprop; finding shortcuts |
| 10 | Attention | RNN, GRU, multi-head attention, transformer | `neural.py` | Direct access to all context |

The lessons form one argument. Each fixes a failure of the one before: counts cannot handle
unseen words, so we smooth; sparse vectors cannot see synonyms, so we learn embeddings; fixed
windows cannot reach far, so we use recurrence and then attention.

## Results worth looking for

All figures come from the lessons as shipped and are deterministic on a given setup. They
measure small teaching datasets, so read them as demonstrations rather than benchmarks.

**A data split can invert your results (lesson 6).** The reviews are near-twins with opposite
labels. Naive Bayes scores **0.13** with random folds, far below chance, and **0.63** once
twins stay on the same side of the split.

**Models find shortcuts (lesson 9).** Predicting a pronoun from a subject seven or more words
back, a 6-word window cannot see the subject, yet it scores 0.77. Slicing the test set shows why:

| Window | All | No distractor noun | With distractor noun |
|---|---|---|---|
| 3 | 0.48 | 0.53 | 0.43 |
| 6 | **0.77** | 0.56 | **1.00** (shortcut) |
| 12 | 1.00 | 1.00 | 1.00 |

The distractor is always the opposite gender, so it leaks the answer.

**Context length decides architecture (`python -m experiments.gap_length`).** Pronoun accuracy
as the dependency stretches (one seed, fixed training budget):

| Extra clauses | Max sentence length | Elman RNN | GRU | Transformer |
|---|---|---|---|---|
| 0 | 13 | 1.00 | 1.00 | 1.00 |
| 3 | 34 | 0.69 | 0.91 | 1.00 |
| 6 | 55 | 0.53 | 0.58 | 1.00 |

The gated RNN outlasts the plain one, and both eventually fall to chance. The transformer also
has more parameters and costs more per step, so this is not a like-for-like contest.

**A surprise in attention maps (lesson 10).** The transformer's prediction flips with the
subject, yet its final-position attention barely lands on the subject. Attention maps are
hints, not explanations.

## Exercises

`exercises/problems.py` has ten functions, one per lesson. `python -m exercises.check` grades
them, mostly against the library's own code. Solutions are in `exercises/solutions.py`.

## Verification

```bash
pip install -r requirements-dev.txt
pytest
```

The 92 tests check the code against independent references where possible:

| Component | Checked against |
|---|---|
| Porter stemmer | NLTK's `ORIGINAL_ALGORITHM` mode, plus examples from the paper |
| Levenshtein distance | NLTK, on random strings |
| TF-IDF | scikit-learn, to numerical precision |
| Naive Bayes | scikit-learn, parameters and scores |
| Viterbi and forward | Brute-force enumeration of every tag sequence |
| Autograd and all models | Numerical gradients |
| BM25 | A hand-computed score |
| Transformer | Causal mask test: changing future tokens leaves earlier outputs unchanged |

## Limits

This code favors clarity over speed and production robustness.

- The datasets are tiny. `STORY`, `DOCUMENTS` and `REVIEWS` are hand-written for this course
  (the reviews are invented examples). The tagging, embedding and agreement corpora come from
  small grammars built so each concept shows up cleanly, which is why several scores reach
  1.00. Natural text is much harder.
- Models train on the CPU with no GPU support. The autograd engine runs in pure NumPy.
- Dense matrices are used where real systems use sparse ones.
- Parsing, named entity recognition, translation and LLM fine-tuning are not covered.

## Where to go next

- Jurafsky & Martin, *Speech and Language Processing*: the standard text, covering most of
  these topics in depth.
- Porter (1980); Mikolov et al. (2013); Levy & Goldberg (2014); Vaswani et al. (2017).
- Once this all makes sense, replace `autograd.py` with PyTorch and train on real data.

## Layout

```
nlpbasics/     the library; docstrings are the course text
lessons/       ten runnable, narrated lessons
exercises/     problems, solutions and auto-checker
experiments/   longer optional experiments
tests/         test suite
```
