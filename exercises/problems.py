"""Exercises, one per lesson. Replace each NotImplementedError, then run:

    python -m exercises.check            check your work
    python -m exercises.check --solutions  confirm the checker against the answers

Everything can be done with the standard library and NumPy. Do not import from
nlpbasics here: the point is to rewrite the core idea yourself.

How to use this file
--------------------
Each function has a comment block above it with four parts:

    Background   why the idea exists and where it shows up in the course
    Task         exactly what to compute, with the formula
    Example      a small worked case you can check by hand
    Watch out    edge cases, conventions the checker relies on, and hints

Try each exercise from its Background and Task first. Read the hints only when
you are stuck. If you are still stuck, the module named under "Read" has the
full library version, and exercises/solutions.py has a short reference answer.
"""
import numpy as np


# =============================================================================
# 1. Character n-grams                                  Lesson 1: text processing
# =============================================================================
# Background
#   A word-level vocabulary has nothing to say about a word it never saw in
#   training. fastText's fix is to also represent each word by its character
#   n-grams: "where" shares pieces with "here", "wherever" and "nowhere", so a
#   new word can borrow meaning from the known words it overlaps with.
#
#   Boundary markers make the edges count. Wrap the word as "<where>" before
#   slicing it. Now "<wh" means "starts with wh" and "re>" means "ends with re".
#   Without the markers, the trigram "her" inside "where" would be the same as
#   the whole word "her". With them, the word is "<her>", a different string.
#
# Task
#   Return every length-n substring of "<" + word + ">", in left-to-right
#   order, keeping duplicates (a repeated n-gram is a real repeat).
#
# Example
#   char_ngrams("cat", 3)    marked = "<cat>"  ->  ["<ca", "cat", "at>"]
#   char_ngrams("where", 3)  ->  ["<wh", "whe", "her", "ere", "re>"]
#   char_ngrams("a", 3)      marked = "<a>"    ->  ["<a>"]
#
# Watch out
#   - A marked string of length L has L - n + 1 windows. That is
#     len(word) + 3 - n, which is zero or less when n > len(word) + 2. Return
#     [] in that case rather than raising an error.
#   - The checker tests "cat" with n=3 and the one-letter word "a" with n=3.
#     The one-letter case is where off-by-one mistakes show up.
#   - Hint: a list comprehension over range(...) with string slicing is enough.
#
# Read: nlpbasics/text.py, lessons/l01_text.py
def char_ngrams(word: str, n: int) -> list[str]:
    """Lesson 1. Character n-grams with boundary markers, as in fastText.
    char_ngrams("cat", 3) == ["<ca", "cat", "at>"]"""
    raise NotImplementedError


# =============================================================================
# 2. Word error rate                                Lesson 2: edit distance
# =============================================================================
# Background
#   Word error rate (WER) is the standard score for speech recognition and is
#   often used for machine translation too. It counts the fewest word-level
#   edits that turn the system's output (the hypothesis) into the correct
#   transcript (the reference):
#
#       substitution  replace one word with another
#       deletion      the hypothesis dropped a reference word
#       insertion     the hypothesis added a word that is not in the reference
#
#   That count is the Levenshtein distance computed over lists of words
#   instead of strings of characters. The algorithm is the same; only the
#   units differ. Dividing by the reference length makes WER comparable across
#   sentences of different lengths.
#
# Task
#   WER = levenshtein(reference, hypothesis) / len(reference)
#
#   Fill a table D where D[i][j] is the distance between the first i words of
#   the reference and the first j words of the hypothesis:
#
#       D[i][0] = i                         delete all i words
#       D[0][j] = j                         insert all j words
#       D[i][j] = min( D[i-1][j]   + 1,     delete reference[i-1]
#                      D[i][j-1]   + 1,     insert hypothesis[j-1]
#                      D[i-1][j-1] + (0 if reference[i-1] == hypothesis[j-1]
#                                     else 1) )   match or substitute
#
#   The distance is D[len(reference)][len(hypothesis)].
#
# Example
#   reference  = "the cat sat on the mat".split()      (6 words)
#   hypothesis = "the cat sat on mat today".split()
#   One cheapest alignment: delete the second "the", match "mat", insert
#   "today". That is 2 edits, so WER = 2 / 6 = 0.333...
#   word_error_rate(["a"], ["a"]) == 0
#
# Watch out
#   - WER is not capped at 1. A hypothesis much longer than the reference can
#     need more edits than the reference has words.
#   - An empty reference makes the division undefined. The checker never
#     passes one, so you may leave that case alone.
#   - Return a float. Python 3's "/" already does that.
#   - Hint: you only ever look at the previous row, so you can keep two lists
#     instead of the whole table. Get the full-table version working first.
#
# Read: nlpbasics/distance.py (edit_matrix, levenshtein), lessons/l02_stemming_distance.py
def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Lesson 2. Word-level edit distance divided by len(reference)."""
    raise NotImplementedError


# =============================================================================
# 3. Most frequent pair                             Lesson 3: Byte Pair Encoding
# =============================================================================
# Background
#   BPE builds a subword vocabulary by repeating one step: find the most
#   common adjacent pair of symbols, give that pair a new id, and replace
#   every occurrence of it. Run this a few thousand times, starting from raw
#   bytes, and frequent words and pieces of words become single tokens while
#   rare words stay split into smaller parts. This exercise is the "find" half
#   of that loop.
#
#   Ties matter more than they seem to. If two runs break ties differently,
#   they learn different merges, and the same text then encodes to different
#   ids. A fixed tie-break keeps training deterministic.
#
# Task
#   Count every adjacent pair (ids[i], ids[i+1]) for i = 0 .. len(ids) - 2.
#   Return the pair with the highest count. If several pairs share the highest
#   count, return the smallest one by ordinary tuple comparison, which compares
#   the first elements and then the second: (1, 9) < (2, 0) < (2, 1).
#
# Example
#   [1, 2, 3, 1, 2]   pairs (1,2) (2,3) (3,1) (1,2)   ->  (1, 2), count 2
#   [5, 6, 1, 2]      pairs (5,6) (6,1) (1,2), each 1 ->  tie, smallest is (1, 2)
#
# Watch out
#   - Count overlapping pairs: [7, 7, 7] contains (7, 7) twice. That is how
#     the library counts. Merging is a separate step that handles overlaps
#     from left to right.
#   - Return a tuple of two ints, not a list.
#   - Fewer than two ids means there are no pairs. The checker does not test
#     that case.
#   - Hint: collections.Counter and zip(ids, ids[1:]) do the counting. For the
#     tie-break, sort by a key: min(counts, key=lambda p: (-counts[p], p)).
#
# Read: nlpbasics/bpe.py (_pair_counts, _merge, BPETokenizer.train), lessons/l03_bpe.py
def most_frequent_pair(ids: list[int]) -> tuple[int, int]:
    """Lesson 3. Most common adjacent pair. Break ties by the smallest pair."""
    raise NotImplementedError


# =============================================================================
# 4. Add-k smoothing                               Lesson 4: n-gram language models
# =============================================================================
# Background
#   A bigram model estimates P(word | previous word) from counts:
#
#       P_MLE(w | c) = count(c, w) / count(c)
#
#   Any pair never seen in training gets probability 0, and one zero makes the
#   probability of a whole test sentence 0 (and its perplexity infinite).
#   Smoothing moves a little probability mass from seen events to unseen ones.
#   Add-k is the simplest method: pretend every word in the vocabulary was
#   seen k extra times after this context. With k = 1 it is called Laplace
#   smoothing. k = 1 is usually far too much for language; values like 0.01 to
#   0.1 work better.
#
# Task
#   P_add-k(w | c) = (count(c, w) + k) / (count(c) + k * V)
#
#   counts         dict mapping each word seen after context c to its count
#   context_total  count(c), the total number of words seen after c
#   vocab_size     V, the number of word types the model can predict
#   word           w, which may be missing from counts (then its count is 0)
#   k              the pseudo-count added to every word
#
#   Why "+ k * V" in the denominator: each of the V words got k extra counts,
#   so the total grew by k * V. That keeps the distribution summing to 1.
#
# Example
#   counts = {"cat": 3, "dog": 1}, context_total = 4, V = 10, k = 1
#   P(cat) = (3 + 1) / (4 + 1 * 10) = 4 / 14 = 0.2857...
#   P(emu) = (0 + 1) / 14 = 0.0714...          unseen, but no longer zero
#
#   Sanity check the checker also runs, with V = 3 and k = 0.5:
#   P(cat) + P(dog) + P(emu) = (3.5 + 1.5 + 0.5) / (4 + 1.5) = 5.5 / 5.5 = 1
#
# Watch out
#   - Use counts.get(word, 0). A missing word is the case smoothing exists for.
#   - Do not recompute context_total from counts. Use the value you are given.
#
# Read: nlpbasics/ngram.py, lessons/l04_ngrams.py
def add_k_bigram(counts: dict, context_total: int, vocab_size: int, word: str, k: float) -> float:
    """Lesson 4. Add-k estimate of P(word | context), where counts maps each word
    to its count after this context and context_total is the count of the context."""
    raise NotImplementedError


# =============================================================================
# 5. BM25 term score                                    Lesson 5: search
# =============================================================================
# Background
#   BM25 is the classic ranking function for keyword search and is still a
#   strong baseline next to neural retrieval. A document's score for a query is
#   the sum, over query terms, of one term's contribution. It improves on
#   TF-IDF with two ideas:
#
#   1. Term frequency saturates. Going from 1 to 2 occurrences of a term is
#      strong evidence of relevance; going from 20 to 21 adds almost nothing.
#      k1 controls how quickly the score levels off. As tf grows, the tf part
#      approaches k1 + 1 instead of growing without limit.
#   2. Long documents are penalized. A long document contains more words by
#      chance, so its tf is discounted relative to the average length.
#      b = 0 turns this off, and b = 1 applies it fully.
#
# Task
#   idf   = ln( 1 + (N - df + 0.5) / (df + 0.5) )
#   score = idf * tf * (k1 + 1) / ( tf + k1 * (1 - b + b * doc_len / avg_len) )
#
#   tf       times the term occurs in this document
#   df       number of documents that contain the term
#   n_docs   N, number of documents in the collection
#   doc_len  length of this document in tokens
#   avg_len  average document length in the collection
#
#   The "1 +" inside the log is the Lucene variant. It keeps idf positive even
#   for a term that appears in more than half the documents. The original
#   formula goes negative there.
#
# Example (the checker's case)
#   docs = [["a","b","a"], ["b","c"], ["c","c","d","e"]], term "a", document 0
#   tf = 2, df = 1, N = 3, doc_len = 3, avg_len = (3 + 2 + 4) / 3 = 3.0
#   idf     = ln(1 + 2.5 / 1.5) = ln(2.667) = 0.9808
#   tf part = 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 1.0)) = 5 / 3.5 = 1.4286
#   score   = 0.9808 * 1.4286 = 1.4012
#
# Watch out
#   - Use the natural log, math.log, so your number matches the library's.
#   - Here doc_len equals avg_len, so the length term is exactly 1. Try a
#     longer document by hand and check that its score goes down.
#
# Read: nlpbasics/vectorspace.py (BM25), lessons/l05_search.py
def bm25_term(tf: int, df: int, n_docs: int, doc_len: int, avg_len: float,
              k1: float = 1.5, b: float = 0.75) -> float:
    """Lesson 5. BM25 score contribution of one term in one document."""
    raise NotImplementedError


# =============================================================================
# 6. F1 from counts                                Lesson 6: classification
# =============================================================================
# Background
#   Accuracy misleads when classes are imbalanced: a spam filter that never
#   flags anything is 99% accurate if 1% of mail is spam. Per-class precision
#   and recall show what is going on:
#
#       precision = tp / (tp + fp)   of the items we flagged, how many were right
#       recall    = tp / (tp + fn)   of the items we should have flagged, how many we got
#
#   F1 is their harmonic mean. A harmonic mean is pulled toward the smaller
#   value, so a classifier cannot score well by doing well on only one of the
#   two.
#
# Task
#   F1 = 2 * P * R / (P + R)
#
#   Substituting the definitions gives a form that uses only counts and never
#   divides by zero unless every count is zero:
#
#       F1 = 2 * tp / (2 * tp + fp + fn)
#
#   Return 0.0 when F1 is undefined.
#
# Example
#   tp = 8, fp = 2, fn = 2   ->  P = 0.8, R = 0.8, F1 = 16 / 20 = 0.8
#   tp = 0, fp = 0, fn = 0   ->  0 / 0, undefined  ->  return 0.0
#   tp = 0, fp = 3, fn = 1   ->  0 / 4 = 0.0 (precision and recall are both 0)
#
# Watch out
#   - If you compute P and R separately, P is 0 / 0 whenever the model
#     predicted nothing (tp + fp = 0). The count form avoids that case.
#   - The checker requires exactly 0.0 for the all-zero case, not nan and not
#     an exception.
#
# Read: nlpbasics/classify.py (classification_report), lessons/l06_classification.py
def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Lesson 6. F1 score. Return 0.0 when it is undefined."""
    raise NotImplementedError


# =============================================================================
# 7. Viterbi decoding                               Lesson 7: HMM tagging
# =============================================================================
# Background
#   A hidden Markov model tagger scores a tag sequence t_1..t_n for words
#   w_1..w_n as
#
#       log P = log_start[t_1] + log_emit[1, t_1]
#             + sum_{i>1} ( log_trans[t_{i-1}, t_i] + log_emit[i, t_i] )
#
#   With T tags there are T^n sequences, far too many to try one by one.
#   Viterbi finds the best one in O(n * T^2) time with dynamic programming,
#   using one fact: the best path ending in tag t at position i must extend
#   the best path ending in some tag s at position i-1. Keep the best score
#   for each tag at each position, plus a pointer to the s it came from, then
#   follow the pointers backwards from the end.
#
#   Everything is in log space, so probabilities multiply by adding logs. That
#   avoids underflow: a product of 50 small probabilities is too small for a
#   float, but a sum of 50 logs is fine.
#
# Task
#   log_start  shape (T,)    log P(first tag = t)
#   log_trans  shape (T, T)  log_trans[s, t] = log P(next tag t | current tag s)
#   log_emit   shape (n, T)  log_emit[i, t] = log P(word i | tag t), precomputed
#
#   V[0, t] = log_start[t] + log_emit[0, t]
#   for i in 1 .. n-1:
#       scores[s, t] = V[i-1, s] + log_trans[s, t]       shape (T, T)
#       back[i, t]   = argmax over s of scores[s, t]
#       V[i, t]      = max over s of scores[s, t] + log_emit[i, t]
#   last = argmax over t of V[n-1, t]
#   then follow back[] from position n-1 to position 1 to recover the path.
#
#   Return the path as a list of n plain ints (tag indices), first word first.
#
# Example
#   With T = 2 and n = 1 there are no transitions: the answer is
#   [argmax(log_start + log_emit[0])]. Get that case right before trying longer
#   sequences.
#
# Watch out
#   - There is no end-of-sentence term here. The library's HMMTagger.viterbi
#     adds log_end at the final step. The checker compares you against a tagger
#     whose log_end is all zeros, so leave it out.
#   - Broadcasting builds scores in one line. V[:, None] has shape (T, 1), so
#     V[:, None] + log_trans adds V[s] to every entry of row s, giving the
#     (T, T) table with rows indexed by the previous tag s.
#   - Mixing up axis=0 and axis=1 is the classic bug. You are maximizing over
#     the previous tag s, which is axis 0 in log_trans[s, t].
#   - The backtrace gives the path last-to-first. Reverse it before returning.
#   - Convert NumPy ints with int(...) so the result is a list of Python ints.
#
# Read: nlpbasics/hmm.py (HMMTagger.viterbi), lessons/l07_tagging.py
def viterbi(log_start: np.ndarray, log_trans: np.ndarray, log_emit: np.ndarray) -> list[int]:
    """Lesson 7. log_start has shape (T,), log_trans (T, T), and log_emit (n, T):
    the emission score of each state at each position. Return the best path."""
    raise NotImplementedError


# =============================================================================
# 8. PPMI matrix                                     Lesson 8: word embeddings
# =============================================================================
# Background
#   "You shall know a word by the company it keeps." Count how often each word
#   w appears near each context word c and you get a co-occurrence matrix whose
#   rows are crude word vectors. Raw counts are dominated by frequent words:
#   "the" is near everything. Pointwise mutual information (PMI) asks a better
#   question: how much more often do w and c occur together than they would if
#   they were independent?
#
#       PMI(w, c) = log( P(w, c) / (P(w) * P(c)) )
#
#   PMI > 0 means they attract each other, PMI < 0 means they avoid each other,
#   and 0 means independent. Negative PMI estimates from sparse counts are
#   mostly noise, so positive PMI (PPMI) clips them to 0. PPMI followed by SVD
#   is a strong classical embedding method, and word2vec has been shown to
#   implicitly factorize a shifted PMI matrix.
#
# Task
#   With C the count matrix, row_w = sum of row w, col_c = sum of column c,
#   and total = sum of all entries:
#
#       P(w, c) / (P(w) P(c))  =  C[w, c] * total / (row_w * col_c)
#
#   PPMI[w, c] = max(0, log of that ratio), with any cell where C[w, c] == 0
#   set to 0.
#
# Example (the checker's matrix)
#   C = [[4, 0, 1],      total = 12
#        [0, 2, 2],      row sums = [5, 4, 3]
#        [1, 2, 0]]      col sums = [5, 4, 3]
#   [0,0]: 4 * 12 / (5 * 5) = 1.92  ->  ln 1.92 = 0.652
#   [0,2]: 1 * 12 / (5 * 3) = 0.80  ->  ln 0.80 < 0, clipped to 0
#   [1,2]: 2 * 12 / (4 * 3) = 2.00  ->  ln 2    = 0.693
#   [0,1]: count 0                  ->  0
#   Full answer, rounded:
#       [[0.652, 0,     0    ],
#        [0,     0.405, 0.693],
#        [0,     0.693, 0    ]]
#
# Watch out
#   - Use the natural log, np.log, to match the library.
#   - Do it with whole-matrix operations. keepdims=True on the sums lets the
#     (n, 1) row sums and (1, m) column sums broadcast to the full shape.
#   - log(0) = -inf, and an all-zero row or column gives 0/0 = nan. Both must
#     end up as 0. np.errstate(divide="ignore", invalid="ignore") silences the
#     warnings, and np.nan_to_num or np.where cleans up the values.
#   - Return a new array. Do not modify counts in place.
#
# Read: nlpbasics/embeddings.py (cooccurrence_matrix, ppmi), lessons/l08_embeddings.py
def ppmi_matrix(counts: np.ndarray) -> np.ndarray:
    """Lesson 8. Positive PMI of a co-occurrence matrix. Zero counts map to 0."""
    raise NotImplementedError


# =============================================================================
# 9. Numerical gradient                           Lesson 9: neural networks
# =============================================================================
# Background
#   Backpropagation computes exact gradients cheaply, but a single sign error
#   in a hand-written backward pass gives gradients that are quietly wrong, and
#   training still runs, just badly. Gradient checking catches this by
#   comparing autograd's answer to a slow numerical estimate from the
#   definition of the derivative. Every op in nlpbasics/autograd.py was tested
#   this way.
#
# Task
#   For each element x_i of x (whatever x's shape is):
#
#       grad_i = ( f(x + h * e_i) - f(x - h * e_i) ) / (2 * h)
#
#   where e_i is 1 at position i and 0 elsewhere. That is: nudge one element
#   up by h, evaluate, nudge it down by h, evaluate, take the slope. Return an
#   array of floats with the same shape as x.
#
#   This is the central difference. Its error shrinks like h^2. The one-sided
#   version (f(x + h) - f(x)) / h only shrinks like h, so it is much less
#   accurate for the same h.
#
# Example
#   f(v) = sum(v ** 3), so the true gradient is 3 * v ** 2
#   x = [1.0, 2.0, -1.0]  ->  approximately [3.0, 12.0, 3.0]
#
# Watch out
#   - The checker verifies that x is unchanged afterwards. If you modify x in
#     place to nudge one element, save the original value and restore it
#     before moving on. Copying x for each nudge also works, but costs more.
#   - Make the output float: np.zeros_like(x, dtype=float). An int input would
#     otherwise truncate every gradient.
#   - np.ndindex(x.shape) visits every index of an array of any shape, so the
#     same code handles vectors and weight matrices.
#   - Smaller h is not always better. Around 1e-8 and below, f(x+h) and f(x-h)
#     agree in almost every digit, and the subtraction loses most of the
#     precision (catastrophic cancellation). 1e-5 is a good choice for float64.
#
# Read: nlpbasics/autograd.py, tests/test_neural.py, lessons/l09_neural.py
def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Lesson 9. Central-difference gradient of scalar f at x. Do not leave x changed."""
    raise NotImplementedError


# =============================================================================
# 10. Causal self-attention                             Lesson 10: transformers
# =============================================================================
# Background
#   Attention lets every position build its output from a weighted mix of
#   other positions. Each position emits a query ("what am I looking for"), a
#   key ("what do I contain") and a value ("what I pass on"). The dot product
#   of a query with each key, turned into probabilities with a softmax, decides
#   how much of each value to take. It works like a dictionary lookup with
#   soft, partial matches.
#
#   A language model predicts the next token, so position t must not see
#   tokens after t, or training would let it read the answer. A causal mask
#   blocks those positions before the softmax, so their weights come out as
#   exactly 0.
#
#   Dividing by sqrt(d) matters. Dot products of random d-dimensional vectors
#   have variance that grows with d. Large scores push the softmax toward a
#   one-hot output, where gradients nearly vanish. Scaling keeps the variance
#   around 1.
#
# Task
#   q, k, v all have shape (T, d).
#
#   scores  = q @ k.T / sqrt(d)               shape (T, T); scores[t, j] = q_t . k_j
#   mask    scores[t, j] = -inf for every j > t   (the part above the diagonal)
#   weights = softmax of each row of scores   each row sums to 1
#   output  = weights @ v                     shape (T, d)
#
# Example
#   Row 0 can attend only to position 0, so its weight vector is
#   [1, 0, 0, ...] and output[0] equals v[0] exactly. The last row attends to
#   every position. Checking output[0] == v[0] is a quick test of the mask.
#
# Watch out
#   - np.triu(np.ones((T, T), dtype=bool), 1) is True exactly where j > t.
#     np.where(mask, -np.inf, scores) applies it.
#   - For a stable softmax, subtract each row's max before np.exp. This does
#     not change the result, but it prevents overflow. A row's max is never
#     -inf, because the diagonal is never masked, so the subtraction is safe.
#   - The library masks with -1e9 instead of -inf. Both give weights that are 0
#     to within the checker's 1e-6 tolerance.
#   - Use d = k.shape[-1], the key dimension. That is the scaling the library
#     uses.
#
# Read: nlpbasics/neural.py (scaled_dot_product_attention, MultiHeadAttention),
#       lessons/l10_attention.py
def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Lesson 10. Scaled dot-product self-attention with a causal mask, all inputs
    of shape (T, d). Each position t may attend only to positions 0..t."""
    raise NotImplementedError
