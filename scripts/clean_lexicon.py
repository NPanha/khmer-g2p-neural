"""Clean annotation artifacts from a Khmer G2P lexicon TSV.

What this fixes
---------------
1. Strips trailing junk characters from IPA values:
   ``.``, ``–`` (en-dash), ``—`` (em-dash), whitespace. These appear in
   a few dozen rows in the wild lexicon and are not phonemes.
2. Normalizes the voiced velar stop: picks ONE of ``g`` (U+0067) vs
   ``ɡ`` (U+0261, IPA script-g) based on whichever is more common in
   your data, and replaces the other.
3. Removes rows where the IPA field is empty (``--drop-empty-ipa``).
4. Removes rows where the word is a standalone Khmer vowel sign or
   punctuation character (``--drop-vowel-punct``).
5. (Optional) De-duplicates exact-duplicate rows (``--dedupe``).

Usage
-----
    # Dry run — show what would change, write nothing
    python scripts/clean_lexicon.py data/lexicon.tsv --dry-run

    # Full clean: drop empty IPA, drop standalone vowel/punct, dedupe
    python scripts/clean_lexicon.py data/lexicon.tsv \\
        --out data/lexicon.clean.tsv \\
        --drop-empty-ipa --drop-vowel-punct --dedupe

Always runs read-only on the input; the cleaned lexicon goes to a new
file so you can diff it against the original.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from khmer_g2p.lexicon import load_tsv  # noqa: E402

# Characters stripped from the end of every IPA string.
TRAILING_JUNK = ".–—·\t "

# Pair: (plain latin g, IPA script g)
G_PLAIN = "g"         # U+0067
G_SCRIPT = "ɡ"   # ɡ — IPA voiced velar stop

# Standalone Khmer vowel signs and punctuation marks that are not real words.
# They produce bad training signal and should be removed from the lexicon.
_STANDALONE_REMOVE = {
    "ី",  # ី   dependent vowel sign II
    "េ",  # េ   dependent vowel sign E
    "ៃ",  # ៃ   dependent vowel sign AI
    "ោ",  # ោ   dependent vowel sign OO
    "។",  # ។   Khmer full stop
    "៕",  # ៕   Khmer sign BARIYOOSAN
}


def _strip_trailing_junk(ipa: str) -> str:
    """Remove trailing '.', '–', '—', whitespace. Preserve middle dots."""
    i = len(ipa)
    while i > 0 and ipa[i - 1] in TRAILING_JUNK:
        i -= 1
    return ipa[:i]


def _pick_g_target(entries) -> str:
    """Pick whichever 'g' is already more common so we minimize edits."""
    c_plain = sum(e[1].count(G_PLAIN) for e in entries)
    c_script = sum(e[1].count(G_SCRIPT) for e in entries)
    return G_PLAIN if c_plain >= c_script else G_SCRIPT


def clean_entries(
    entries: List[Tuple[str, str]],
    drop_empty_ipa: bool = False,
    drop_vowel_punct: bool = False,
):
    target_g = _pick_g_target(entries)
    other_g = G_SCRIPT if target_g == G_PLAIN else G_PLAIN

    cleaned: List[Tuple[str, str]] = []
    changes = {
        "trailing_stripped": 0,
        "g_normalized": 0,
        "empty_ipa_dropped": 0,
        "vowel_punct_dropped": 0,
        "unchanged": 0,
    }
    samples = {"trailing": [], "g": []}

    for word, ipa in entries:
        if drop_empty_ipa and not ipa.strip():
            changes["empty_ipa_dropped"] += 1
            continue
        if drop_vowel_punct and word in _STANDALONE_REMOVE:
            changes["vowel_punct_dropped"] += 1
            continue

        new = ipa
        if any(new.endswith(c) for c in TRAILING_JUNK):
            stripped = _strip_trailing_junk(new)
            if stripped != new:
                changes["trailing_stripped"] += 1
                if len(samples["trailing"]) < 10:
                    samples["trailing"].append((word, ipa, stripped))
                new = stripped
        if other_g in new:
            swapped = new.replace(other_g, target_g)
            if swapped != new:
                changes["g_normalized"] += 1
                if len(samples["g"]) < 10:
                    samples["g"].append((word, ipa, swapped))
                new = swapped
        if new == ipa:
            changes["unchanged"] += 1
        cleaned.append((word, new))

    return cleaned, changes, samples, target_g, other_g


def _dedupe(entries: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], int]:
    seen = set()
    out = []
    removed = 0
    for w, p in entries:
        key = (w, p)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append((w, p))
    return out, removed


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clean_lexicon",
                                  description=__doc__.splitlines()[0])
    ap.add_argument("tsv", type=Path, help="Input lexicon TSV")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the cleaned TSV (required unless --dry-run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show changes and write nothing.")
    ap.add_argument("--dedupe", action="store_true",
                    help="Drop duplicate (word, ipa) rows after cleanup.")
    ap.add_argument("--drop-empty-ipa", action="store_true",
                    help="Remove rows where the IPA field is empty.")
    ap.add_argument("--drop-vowel-punct", action="store_true",
                    help="Remove rows where the word is a standalone Khmer "
                         "vowel sign or punctuation mark.")
    args = ap.parse_args(argv)

    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run is set")

    entries = load_tsv(args.tsv)
    cleaned, changes, samples, target_g, other_g = clean_entries(
        entries,
        drop_empty_ipa=args.drop_empty_ipa,
        drop_vowel_punct=args.drop_vowel_punct,
    )

    print(f"Input:                     {args.tsv}")
    print(f"Rows loaded:               {len(entries)}")
    print(f"Chose target voiced velar: {target_g!r}  "
          f"(replacing {other_g!r} → {target_g!r})")
    print()
    if args.drop_empty_ipa:
        print(f"Rows with empty IPA dropped:      {changes['empty_ipa_dropped']}")
    if args.drop_vowel_punct:
        print(f"Rows with vowel/punct word dropped:{changes['vowel_punct_dropped']}")
    print(f"Rows with trailing junk stripped: {changes['trailing_stripped']}")
    for w, old, new in samples["trailing"]:
        print(f"    {w}  IPA: {old!r}  →  {new!r}")
    print()
    print(f"Rows with g / ɡ normalized:       {changes['g_normalized']}")
    for w, old, new in samples["g"]:
        print(f"    {w}  IPA: {old!r}  →  {new!r}")

    if args.dedupe:
        cleaned, removed = _dedupe(cleaned)
        print(f"\nDuplicate rows removed:          {removed}")

    print(f"\nTotal rows affected: "
          f"{changes['trailing_stripped'] + changes['g_normalized'] + changes['empty_ipa_dropped'] + changes['vowel_punct_dropped']}")
    print(f"Total rows unchanged: {changes['unchanged']}")
    print(f"Final rows: {len(cleaned)}")

    if args.dry_run:
        print("\n(dry run — no file written)")
        return 0

    assert args.out is not None
    with args.out.open("w", encoding="utf-8") as f:
        f.write("word\tipa\n")
        for w, p in cleaned:
            f.write(f"{w}\t{p}\n")
    print(f"\nWrote cleaned lexicon: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
