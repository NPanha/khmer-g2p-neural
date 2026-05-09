"""Evaluation metrics for Khmer G2P.

Core metrics (used by the training loop):
    - edit_distance        : Levenshtein distance between two sequences.
    - per                  : Phone Error Rate (edit-distance-based).
    - wer                  : Word Error Rate (exact-match fraction).

Extended metrics (used by the notebook / offline eval):
    - accuracy             : 1 − WER (exact-match rate).
    - per_tokens           : PER over phoneme tokens (use with a tokenizer).
    - per_per_word         : Per-example PER (for bucketed analysis).
    - length_bucket_stats  : Accuracy / PER by word length bucket.
    - phoneme_confusion    : Confusion matrix from edit-distance alignments.
    - bleu                 : Corpus-level BLEU (with optional n-gram weights).
    - topk_exact_match     : Top-k exact match given a list of k candidates.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Core: edit distance + error rates
# ---------------------------------------------------------------------------


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[lb]


def per(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Phone Error Rate, character-level over the raw strings."""
    if not predictions:
        return 0.0
    total_edits = 0
    total_len = 0
    for p, r in zip(predictions, references):
        total_edits += edit_distance(list(p), list(r))
        total_len += max(1, len(r))
    return total_edits / total_len


def per_tokens(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenize: Callable[[str], List[str]],
) -> float:
    """PER computed over phoneme tokens instead of characters.

    Pass ``phoneme_tokenizer.tokenize`` to get phoneme-level PER.
    """
    if not predictions:
        return 0.0
    total_edits = 0
    total_len = 0
    for p, r in zip(predictions, references):
        a, b = tokenize(p), tokenize(r)
        total_edits += edit_distance(a, b)
        total_len += max(1, len(b))
    return total_edits / total_len


def wer(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Word Error Rate = fraction of whole-sequence mismatches."""
    if not predictions:
        return 0.0
    wrong = sum(1 for p, r in zip(predictions, references) if p != r)
    return wrong / len(predictions)


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact-match accuracy = 1 − WER."""
    return 1.0 - wer(predictions, references)


def per_per_word(
    predictions: Sequence[str],
    references: Sequence[str],
) -> List[float]:
    """Per-example PER — returns one float per (pred, ref) pair."""
    out: List[float] = []
    for p, r in zip(predictions, references):
        d = edit_distance(list(p), list(r))
        out.append(d / max(1, len(r)))
    return out


# ---------------------------------------------------------------------------
# Bucketed analysis
# ---------------------------------------------------------------------------


def length_bucket_stats(
    words: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    bucket_edges: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 15, 1_000_000),
) -> List[Dict[str, float]]:
    """Accuracy & PER bucketed by source-word length.

    Returns a list of dicts: ``{lo, hi, n, accuracy, per}``.
    """
    buckets: List[Dict[str, list]] = [
        {"lo": lo, "hi": hi, "items": []}
        for lo, hi in zip(bucket_edges, bucket_edges[1:])
    ]
    for w, p, r in zip(words, predictions, references):
        L = len(w)
        for b in buckets:
            if b["lo"] <= L < b["hi"]:
                b["items"].append((p, r))
                break
    out: List[Dict[str, float]] = []
    for b in buckets:
        items = b["items"]
        if not items:
            out.append({"lo": b["lo"], "hi": b["hi"], "n": 0,
                        "accuracy": float("nan"), "per": float("nan")})
            continue
        preds = [p for p, _ in items]
        refs = [r for _, r in items]
        out.append({
            "lo": b["lo"], "hi": b["hi"], "n": len(items),
            "accuracy": accuracy(preds, refs),
            "per": per(preds, refs),
        })
    return out


# ---------------------------------------------------------------------------
# Phoneme confusion matrix (from edit-distance alignments)
# ---------------------------------------------------------------------------


def _align(a: Sequence[str], b: Sequence[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Align two token sequences via Levenshtein backtrace.

    Returns a list of (pred_token, ref_token) pairs. None on either side
    means insertion or deletion.
    """
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    # Backtrace
    out: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((a[i - 1], None)); i -= 1            # insertion in pred
        else:
            out.append((None, b[j - 1])); j -= 1            # deletion from pred
    out.reverse()
    return out


def phoneme_confusion(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenize: Callable[[str], List[str]],
    include_correct: bool = False,
) -> Counter:
    """Counter of (pred_token, ref_token) mismatches across the whole set.

    Keys: (pred_phoneme_or_None, ref_phoneme_or_None). ``None`` on the
    pred side means "reference had a phoneme the model skipped"; ``None``
    on the ref side means "model hallucinated a phoneme".

    Use ``include_correct=True`` to also count correct alignments (useful
    for computing per-phoneme precision/recall).
    """
    counts: Counter = Counter()
    for p, r in zip(predictions, references):
        a, b = tokenize(p), tokenize(r)
        for pa, pb in _align(a, b):
            if pa == pb and not include_correct:
                continue
            counts[(pa, pb)] += 1
    return counts


def phoneme_precision_recall(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenize: Callable[[str], List[str]],
) -> Dict[str, Dict[str, float]]:
    """Per-phoneme precision, recall, F1.

    Precision = (correct emissions of X) / (all emissions of X)
    Recall    = (correct emissions of X) / (all references of X)
    """
    tp: Counter = Counter()
    emitted: Counter = Counter()
    referenced: Counter = Counter()
    for p, r in zip(predictions, references):
        a, b = tokenize(p), tokenize(r)
        for pa, pb in _align(a, b):
            if pa is not None:
                emitted[pa] += 1
            if pb is not None:
                referenced[pb] += 1
            if pa is not None and pa == pb:
                tp[pa] += 1
    keys = set(emitted) | set(referenced)
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        e = emitted[k]
        ref = referenced[k]
        t = tp[k]
        prec = t / e if e else 0.0
        rec = t / ref if ref else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        out[k] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": ref, "emitted": e}
    return out


# ---------------------------------------------------------------------------
# BLEU (character / phoneme)
# ---------------------------------------------------------------------------


def _ngrams(seq: Sequence[str], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def bleu(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenize: Optional[Callable[[str], List[str]]] = None,
    max_n: int = 4,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Corpus-level BLEU. Tokenization defaults to character-level.

    Pass ``tokenize=phoneme_tokenizer.tokenize`` for phoneme-BLEU.
    """
    if not predictions:
        return 0.0
    weights = list(weights) if weights else [1.0 / max_n] * max_n
    tok = tokenize if tokenize is not None else list

    match_counts = [0] * max_n
    total_counts = [0] * max_n
    pred_len_total = 0
    ref_len_total = 0
    for pred_str, ref_str in zip(predictions, references):
        p_tok = tok(pred_str)
        r_tok = tok(ref_str)
        pred_len_total += len(p_tok)
        ref_len_total += len(r_tok)
        for i, n in enumerate(range(1, max_n + 1)):
            p_ng = _ngrams(p_tok, n)
            r_ng = _ngrams(r_tok, n)
            total_counts[i] += max(0, len(p_tok) - n + 1)
            for ng, cnt in p_ng.items():
                match_counts[i] += min(cnt, r_ng.get(ng, 0))

    # Precisions with tiny smoothing for zero counts
    precisions = []
    for m, t in zip(match_counts, total_counts):
        if t == 0:
            precisions.append(0.0)
        else:
            precisions.append((m + 1e-9) / (t + 1e-9))

    if min(precisions) <= 0:
        return 0.0

    log_avg = sum(w * math.log(p) for w, p in zip(weights, precisions))
    # Brevity penalty
    if pred_len_total > ref_len_total:
        bp = 1.0
    elif pred_len_total == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len_total / pred_len_total)
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# Top-k exact match
# ---------------------------------------------------------------------------


def topk_exact_match(
    candidates_per_word: Sequence[Sequence[str]],
    references: Sequence[str],
    k: int = 5,
) -> float:
    """Fraction of examples where the reference IPA is within the top-k predictions."""
    if not candidates_per_word:
        return 0.0
    hits = 0
    for cands, ref in zip(candidates_per_word, references):
        if ref in list(cands)[:k]:
            hits += 1
    return hits / len(candidates_per_word)
