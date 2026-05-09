"""Audit a Khmer G2P lexicon TSV for annotation inconsistencies.

Usage:
    python scripts/audit_lexicon.py data/lexicon.tsv
    python scripts/audit_lexicon.py data/lexicon.tsv --top 40 --out audit_report.txt

What this looks for
-------------------
1. **Duplicate rows** — identical (word, ipa) pair appearing more than once.
2. **Conflicting variants** — one word mapped to many different IPAs. Some
   of this is real (free variation, dialectal variants), but if a word has
   more than, say, 3 variants it's usually a data entry issue.
3. **Leading/trailing whitespace** drift — rows where word or IPA has
   stray leading/trailing whitespace that would mis-merge.
4. **Whitespace-only or empty fields**.
5. **Stress-mark inconsistency** — same word, same consonants/vowels, but
   differs only in presence/position of the 'ˈ' stress mark.
6. **Rare phonemes** — phonemes that appear in fewer than N lexicon rows.
   These are either typos or legitimately rare phonemes worth double-checking.
7. **Unusually long IPAs** — top-K longest pronunciations, since runaway
   IPA strings are often copy/paste accidents.
8. **Unknown phonemes** — phonemes not in the reference Khmer IPA inventory.

Exit code
---------
Always 0 (this is a report tool, not a linter). The point is to give you
a punch-list, not gate CI.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Make the package importable when running this as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from khmer_g2p.lexicon import load_tsv  # noqa: E402
from khmer_g2p.neural.phoneme_tokenizer import tokenize, flag_unknown  # noqa: E402


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def audit(
    path: Path,
    top: int = 20,
    max_variants_warn: int = 3,
    rare_phoneme_threshold: int = 3,
) -> List[str]:
    """Run all diagnostics and return a list of report lines."""
    entries = load_tsv(path)
    lines: List[str] = []
    lines.append(f"Lexicon: {path}")
    lines.append(f"Total rows loaded: {len(entries)}")

    # ---- 1. Duplicate exact rows --------------------------------------------
    row_counts = Counter((w, p) for w, p in entries)
    dupes = [(w, p, c) for (w, p), c in row_counts.items() if c > 1]
    lines.append(_section(f"1. Exact-duplicate rows: {len(dupes)}"))
    for w, p, c in sorted(dupes, key=lambda x: -x[2])[:top]:
        lines.append(f"  [{c}×] {w}\t{p}")

    # ---- 2. Conflicting variants --------------------------------------------
    by_word: Dict[str, List[str]] = defaultdict(list)
    for w, p in entries:
        by_word[w].append(p)
    many_variants = [(w, ps) for w, ps in by_word.items() if len(set(ps)) > max_variants_warn]
    lines.append(_section(
        f"2. Words with >{max_variants_warn} distinct pronunciations: {len(many_variants)}"
    ))
    for w, ps in sorted(many_variants, key=lambda x: -len(set(x[1])))[:top]:
        uniq = list(dict.fromkeys(ps))  # preserve order, unique
        lines.append(f"  {w}  ({len(uniq)} variants)")
        for v in uniq:
            lines.append(f"      {v}")

    # ---- 3. Whitespace drift in raw file ------------------------------------
    # (load_tsv already strips, so re-read the raw file to catch this.)
    whitespace_issues: List[Tuple[int, str, str]] = []
    empty_fields: List[Tuple[int, str]] = []
    with path.open(encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.strip().split(None, 1)
            if len(parts) < 2:
                empty_fields.append((lineno, line))
                continue
            w_raw, p_raw = parts[0], parts[1]
            if w_raw != w_raw.strip() or p_raw != p_raw.strip():
                whitespace_issues.append((lineno, w_raw, p_raw))

    lines.append(_section(f"3. Rows with leading/trailing whitespace: {len(whitespace_issues)}"))
    for lineno, w, p in whitespace_issues[:top]:
        lines.append(f"  line {lineno}: word={w!r}  ipa={p!r}")

    lines.append(_section(f"4. Rows with empty/missing fields: {len(empty_fields)}"))
    for lineno, raw in empty_fields[:top]:
        lines.append(f"  line {lineno}: {raw!r}")

    # ---- 5. Stress-only disagreement ----------------------------------------
    # Two variants of a word differ only in stress marks.
    stress_only: List[Tuple[str, List[str]]] = []
    for w, ps in by_word.items():
        stripped = {p.replace("ˈ", "") for p in ps}
        if len(stripped) == 1 and len(set(ps)) > 1:
            stress_only.append((w, list(dict.fromkeys(ps))))
    lines.append(_section(
        f"5. Words whose variants differ only in stress 'ˈ': {len(stress_only)}"
    ))
    for w, ps in stress_only[:top]:
        lines.append(f"  {w}  →  {' | '.join(ps)}")

    # ---- 6. Rare & unknown phonemes -----------------------------------------
    phoneme_counts: Counter = Counter()
    all_tokens: set = set()
    for _, p in entries:
        toks = tokenize(p)
        all_tokens.update(toks)
        phoneme_counts.update(toks)

    rare = sorted(
        [(t, c) for t, c in phoneme_counts.items() if c <= rare_phoneme_threshold],
        key=lambda x: (x[1], x[0]),
    )
    lines.append(_section(
        f"6. Rare phonemes (≤{rare_phoneme_threshold} occurrences): {len(rare)}"
    ))
    for tok, c in rare[:top * 2]:  # show more here — this is usually the gold
        examples = [w + "\t" + p for w, p in entries if tok in tokenize(p)][:3]
        lines.append(f"  {tok!r} ({c}×)   e.g. {examples[0] if examples else ''}")

    unknown = [t for t in all_tokens if t and t not in {" "} and flag_unknown([t])]
    lines.append(_section(
        f"7. Char-level tokens outside the reference inventory: {len(unknown)}"
    ))
    for tok in sorted(unknown):
        lines.append(f"  {tok!r}  (count={phoneme_counts[tok]})")

    # ---- 7b. Unit-level check: space-separated phoneme atoms ----------------
    # The training-data lexicon uses space-separated notation ('ph', 'aa', 'ie'…).
    # The char-level tokenizer splits these into individual chars, so the check
    # above misses whole-unit typos (e.g. 'pw' for 'ph'). Check units directly.
    from khmer_g2p.neural.phoneme_tokenizer import KHMER_IPA_INVENTORY as _INV
    unit_counts: Counter = Counter()
    for _, p in entries:
        for unit in p.split():
            unit_counts[unit] += 1
    unknown_units = [u for u in unit_counts if u not in _INV]
    lines.append(_section(
        f"7b. Space-separated phoneme units outside inventory: {len(unknown_units)}"
    ))
    for u in sorted(unknown_units, key=lambda x: -unit_counts[x])[:top]:
        examples = [w for w, p in entries if u in p.split()][:3]
        lines.append(f"  {u!r} ({unit_counts[u]}×)  e.g. {examples}")

    # ---- 8. Longest IPA strings ---------------------------------------------
    longest = sorted(entries, key=lambda x: -len(x[1]))[:top]
    lines.append(_section(f"8. Longest IPA strings (top {top})"))
    for w, p in longest:
        lines.append(f"  len={len(p):>3}  {w}\t{p}")

    # ---- Summary ------------------------------------------------------------
    lines.append(_section("Summary"))
    lines.append(f"  Unique words:             {len(by_word)}")
    lines.append(f"  Distinct phoneme tokens:  {len(phoneme_counts)}")
    lines.append(f"  Exact duplicates:         {len(dupes)}")
    lines.append(f"  Words with >{max_variants_warn} variants:  {len(many_variants)}")
    lines.append(f"  Whitespace issues:        {len(whitespace_issues)}")
    lines.append(f"  Stress-only disagree:     {len(stress_only)}")
    lines.append(f"  Rare phonemes:            {len(rare)}")
    lines.append(f"  Unknown phonemes:         {len(unknown)}")

    return lines


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_lexicon",
        description="Audit a Khmer G2P lexicon TSV for data-quality issues.",
    )
    parser.add_argument("tsv", type=Path, help="Path to lexicon.tsv")
    parser.add_argument("--top", type=int, default=20,
                        help="Max rows to show per issue category (default: 20).")
    parser.add_argument("--max-variants", type=int, default=3,
                        help="Flag words with more than this many distinct pronunciations.")
    parser.add_argument("--rare-threshold", type=int, default=3,
                        help="Flag phonemes occurring ≤ this many times.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write report to a file in addition to stdout.")
    args = parser.parse_args(argv)

    report = audit(
        args.tsv,
        top=args.top,
        max_variants_warn=args.max_variants,
        rare_phoneme_threshold=args.rare_threshold,
    )
    text = "\n".join(report)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
