"""Phonemize a TTS metadata file with the Khmer G2P pipeline.

Reads a TTS-style metadata file (LJSpeech / StyleTTS2 conventions —
``audio|text|...`` with ``|``, or TSV / CSV) and writes a copy where the
text column has been replaced with phonemes from the trained G2P model.

Examples
--------

    # LJSpeech-style: 'wav|text|speaker' separated by '|'
    python scripts/phonemize_metadata.py \\
        --ckpt checkpoints_v04/best.pt \\
        --lexicon data/lexicon.tsv \\
        --in train_list.txt --out train_list.phon.txt \\
        --sep '|' --text-col 1

    # TSV with a header row
    python scripts/phonemize_metadata.py \\
        --ckpt checkpoints_v04/best.pt \\
        --lexicon data/lexicon.tsv \\
        --in metadata.tsv --out metadata.phon.tsv \\
        --sep '\\t' --text-col 1 --has-header

    # Plain text, one sentence per line
    python scripts/phonemize_metadata.py \\
        --ckpt checkpoints_v04/best.pt \\
        --lexicon data/lexicon.tsv \\
        --in sentences.txt --out sentences.phon.txt --plain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Make the script runnable without `pip install -e .` by injecting src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from khmer_g2p.pipeline import KhmerG2PPipeline, PipelineConfig  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):  # minimal fallback
        return it


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phonemize a TTS metadata file using khmer-g2p-neural.",
    )
    p.add_argument("--ckpt", required=True, help="Path to best.pt")
    p.add_argument("--lexicon", required=True, help="Path to lexicon TSV")
    p.add_argument("--in", dest="inp", required=True, help="Input metadata file")
    p.add_argument("--out", required=True, help="Output file")

    p.add_argument(
        "--sep",
        default="|",
        help="Field separator. Use '\\t' for TSV. Default: '|'.",
    )
    p.add_argument(
        "--text-col",
        type=int,
        default=1,
        help="0-indexed column that contains the text to phonemize. Default: 1.",
    )
    p.add_argument(
        "--has-header",
        action="store_true",
        help="If set, copy the first line through unchanged.",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Treat the input as plain text, one sentence per line "
            "(ignores --sep and --text-col)."
        ),
    )

    p.add_argument("--beam", type=int, default=4, help="Neural beam width. Default: 4.")
    p.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="GNMT-style beam length penalty. Default: 0.6.",
    )
    p.add_argument(
        "--word-boundary",
        default="|",
        help="Word boundary token in the output. Default: '|'.",
    )
    p.add_argument(
        "--drop-non-khmer",
        action="store_true",
        help="Drop punctuation / ASCII instead of passing it through.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' or 'cuda'. Default: auto.",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=0,
        help="Print hit-rate / cache stats every N rows (0 = off).",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    sep = "\t" if args.sep == "\\t" else args.sep
    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        word_boundary=args.word_boundary,
        beam=args.beam,
        length_penalty=args.length_penalty,
        keep_non_khmer=not args.drop_non_khmer,
    )
    print(f"[load] ckpt={args.ckpt}  lexicon={args.lexicon}", file=sys.stderr)
    pipe = KhmerG2PPipeline.from_paths(
        ckpt=args.ckpt,
        lexicon=args.lexicon,
        device=args.device,
        config=cfg,
    )
    print(
        f"[ready] lexicon size={len(pipe.g2p)}  segmenter vocab={len(pipe.segmenter)}",
        file=sys.stderr,
    )

    n_in = n_out = n_skipped = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="\n") as fout:
        if args.has_header and not args.plain:
            header = fin.readline()
            if header:
                fout.write(header if header.endswith("\n") else header + "\n")

        for raw in tqdm(fin, desc="phonemize"):
            n_in += 1
            line = raw.rstrip("\r\n")
            if not line:
                fout.write("\n")
                continue

            if args.plain:
                phon = pipe.phonemize(line)
                fout.write(phon + "\n")
                n_out += 1
            else:
                fields = line.split(sep)
                if args.text_col >= len(fields):
                    print(
                        f"[warn] line {n_in}: only {len(fields)} fields, "
                        f"text-col={args.text_col} out of range — passed through",
                        file=sys.stderr,
                    )
                    fout.write(line + "\n")
                    n_skipped += 1
                    continue
                fields[args.text_col] = pipe.phonemize(fields[args.text_col])
                fout.write(sep.join(fields) + "\n")
                n_out += 1

            if args.report_every and n_in % args.report_every == 0:
                print(
                    f"[stat] rows={n_in} hit_rate={pipe.hit_rate:.3f}",
                    file=sys.stderr,
                )

    print(
        f"[done] in={n_in} out={n_out} skipped={n_skipped} "
        f"lexicon_hit_rate={pipe.hit_rate:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
