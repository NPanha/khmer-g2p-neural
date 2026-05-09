"""Interactive live tester for khmer-g2p-neural.

Usage:
    python scripts/live_test.py --ckpt checkpoints_v03/best.pt
    python scripts/live_test.py --ckpt checkpoints_v03/best.pt --lexicon data/lexicon.tsv
    python scripts/live_test.py --ckpt checkpoints_v03/best.pt --beam 4

Type Khmer text at the '>>' prompt and press Enter to get the IPA output.

Input modes (auto-detected):
  - Single word        → one model call → one IPA string.
  - Whitespace-joined  → each token converted independently, results joined
                          with spaces. Non-Khmer tokens are passed through.
  - Run-on Khmer text  → if --lexicon is given, greedy longest-match
                          segmentation is used to split into words; otherwise
                          the whole run is sent to the model as one "word".

REPL commands (start with ':'):
    :beam N        set beam size (1 = greedy)
    :greedy        shortcut for ':beam 1'
    :compare       show greedy + beam side-by-side for each input
    :seg on|off    toggle lexicon-based segmentation
    :time on|off   toggle per-query timing
    :show-norm     toggle showing the NFC-normalized input
    :help          print this help
    :quit / :q     exit
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

# Local imports — delayed so --help works without torch in some envs.
def _load_g2p_module():
    from khmer_g2p.neural.infer import NeuralG2P  # noqa: F401
    from khmer_g2p.normalizer import normalize    # noqa: F401
    from khmer_g2p.segmenter import Segmenter     # noqa: F401
    from khmer_g2p.lexicon import load_tsv        # noqa: F401
    import khmer_g2p.neural.infer as _i
    import khmer_g2p.normalizer as _n
    import khmer_g2p.segmenter as _s
    import khmer_g2p.lexicon as _lx
    return _i.NeuralG2P, _n.normalize, _s.Segmenter, _lx.load_tsv


# A Khmer character run = any Unicode chars in the Khmer block (U+1780..U+17FF).
_KHMER_RUN_RE = re.compile(r"[\u1780-\u17FF]+")
_ANY_KHMER_RE = re.compile(r"[\u1780-\u17FF]")


def _is_khmer_token(tok: str) -> bool:
    return bool(_ANY_KHMER_RE.search(tok))


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def convert_text(
    text: str,
    g2p,
    beam: int,
    segmenter: Optional[object] = None,
    normalize_fn=None,
) -> str:
    """Convert a user input line to an IPA string.

    Strategy:
        - Split on whitespace first.
        - For each whitespace-separated token:
            * If it contains no Khmer, emit verbatim.
            * Else if a segmenter is provided, segment each maximal Khmer run
              into words and convert each word.
            * Else, send the full Khmer-containing token to the model as-is.
    """
    tokens = text.strip().split()
    if not tokens:
        return ""

    out_parts: List[str] = []
    for tok in tokens:
        if not _is_khmer_token(tok):
            out_parts.append(tok)  # pass through ASCII, punctuation, numbers
            continue

        if segmenter is not None:
            # Split non-Khmer runs from Khmer runs inside this token so we
            # don't lose e.g. trailing "?" or digits that stuck to a word.
            pieces: List[str] = []
            last = 0
            for m in _KHMER_RUN_RE.finditer(tok):
                if m.start() > last:
                    pieces.append(tok[last:m.start()])  # non-khmer chunk
                km = m.group()
                km_norm = normalize_fn(km) if normalize_fn else km
                words = segmenter.segment(km_norm)
                pieces.append(" ".join(g2p.convert(w, beam=beam) for w in words))
                last = m.end()
            if last < len(tok):
                pieces.append(tok[last:])
            out_parts.append("".join(pieces))
        else:
            # No segmenter: hand the token to the model as one word. If it has
            # trailing punctuation, we split it off so the model sees clean input.
            m = re.match(r"^(\W*)([\u1780-\u17FF].*?)(\W*)$", tok, flags=re.UNICODE)
            if m:
                lead, core, trail = m.group(1), m.group(2), m.group(3)
            else:
                lead, core, trail = "", tok, ""
            ipa = g2p.convert(core, beam=beam)
            out_parts.append(f"{lead}{ipa}{trail}")

    return " ".join(out_parts)


def compare(text: str, g2p, segmenter, normalize_fn, beams=(1, 4)) -> List[str]:
    rows = []
    for b in beams:
        label = "greedy" if b == 1 else f"beam={b}"
        ipa = convert_text(text, g2p, beam=b,
                           segmenter=segmenter, normalize_fn=normalize_fn)
        rows.append(f"  {label:>8}  →  {ipa}")
    return rows


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP = """Commands:
    :beam N        set beam size (1 = greedy)
    :greedy        shortcut for ':beam 1'
    :compare       toggle greedy + beam side-by-side output
    :seg on|off    toggle lexicon-based segmentation (requires --lexicon)
    :time on|off   toggle per-query timing
    :show-norm     toggle showing the NFC-normalized input
    :help          print this help
    :quit / :q     exit
"""


def repl(
    g2p,
    normalize_fn,
    segmenter,
    default_beam: int,
    default_seg: bool,
) -> int:
    beam = default_beam
    use_seg = default_seg and segmenter is not None
    compare_mode = False
    show_time = True
    show_norm = False

    print()
    print("khmer-g2p-neural — live test")
    print(f"  device   : {g2p.device}")
    print(f"  beam     : {beam}  (':beam N' to change, ':greedy' for 1)")
    print(f"  segment  : {'on' if use_seg else 'off'}"
          + ("" if segmenter else "  (no lexicon provided)"))
    print("  Type Khmer text and press Enter.  ':help' for commands, ':quit' to exit.\n")

    try:
        while True:
            try:
                line = input(">> ").strip()
            except EOFError:
                print()
                return 0
            if not line:
                continue

            # --- commands -----------------------------------------------------
            if line.startswith(":"):
                parts = line.split()
                cmd = parts[0].lower()
                if cmd in (":quit", ":q", ":exit"):
                    return 0
                if cmd == ":help":
                    print(HELP)
                    continue
                if cmd == ":greedy":
                    beam = 1
                    print("  beam = 1 (greedy)")
                    continue
                if cmd == ":beam":
                    if len(parts) != 2 or not parts[1].isdigit():
                        print("  usage: :beam N    (positive integer)")
                        continue
                    beam = max(1, int(parts[1]))
                    print(f"  beam = {beam}")
                    continue
                if cmd == ":compare":
                    compare_mode = not compare_mode
                    print(f"  compare mode = {'on' if compare_mode else 'off'}")
                    continue
                if cmd == ":seg":
                    if segmenter is None:
                        print("  no lexicon loaded — pass --lexicon to enable.")
                        continue
                    if len(parts) == 2 and parts[1] in {"on", "off"}:
                        use_seg = (parts[1] == "on")
                    else:
                        use_seg = not use_seg
                    print(f"  segmentation = {'on' if use_seg else 'off'}")
                    continue
                if cmd == ":time":
                    if len(parts) == 2 and parts[1] in {"on", "off"}:
                        show_time = (parts[1] == "on")
                    else:
                        show_time = not show_time
                    print(f"  timing = {'on' if show_time else 'off'}")
                    continue
                if cmd == ":show-norm":
                    show_norm = not show_norm
                    print(f"  show normalized = {'on' if show_norm else 'off'}")
                    continue
                print(f"  unknown command: {cmd!r}   (':help' for list)")
                continue

            # --- normal input -------------------------------------------------
            if show_norm:
                print(f"  nfc  : {normalize_fn(line)}")

            seg = segmenter if use_seg else None
            t0 = time.time()
            try:
                if compare_mode:
                    rows = compare(line, g2p, seg, normalize_fn)
                    for r in rows:
                        print(r)
                else:
                    ipa = convert_text(line, g2p, beam=beam,
                                       segmenter=seg, normalize_fn=normalize_fn)
                    print(f"  ipa  : {ipa}")
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}")
                continue
            dt = (time.time() - t0) * 1000.0
            if show_time:
                print(f"  time : {dt:.1f} ms")
    except KeyboardInterrupt:
        print()
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="Path to best.pt or last.pt from training.")
    ap.add_argument("--lexicon", type=Path, default=None,
                    help="Optional TSV lexicon to enable sentence segmentation.")
    ap.add_argument("--beam", type=int, default=1,
                    help="Default beam size (1 = greedy).")
    ap.add_argument("--seg", choices=["auto", "on", "off"], default="auto",
                    help="Default segmentation mode. 'auto' = on iff --lexicon given.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-len", type=int, default=128,
                    help="Max decode length per word.")
    ap.add_argument("text", nargs="*",
                    help="Optional: if given, run once and exit (non-interactive).")
    args = ap.parse_args(argv)

    if not args.ckpt.exists():
        print(f"checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 2

    NeuralG2P, normalize_fn, Segmenter, load_tsv = _load_g2p_module()

    device = None if args.device == "auto" else args.device
    print(f"Loading checkpoint: {args.ckpt}")
    g2p = NeuralG2P.from_checkpoint(args.ckpt, device=device, max_len=args.max_len)

    segmenter = None
    if args.lexicon is not None:
        if not args.lexicon.exists():
            print(f"lexicon not found: {args.lexicon}", file=sys.stderr)
            return 2
        pairs = load_tsv(str(args.lexicon))
        words = [w for w, _ in pairs]
        segmenter = Segmenter(words)
        print(f"Loaded segmenter over {len(segmenter):,} unique words "
              f"(max_word_len={segmenter.max_word_len})")

    default_seg = (
        segmenter is not None if args.seg == "auto" else args.seg == "on"
    )

    # One-shot mode — if text was passed on the CLI, convert it and exit.
    if args.text:
        line = " ".join(args.text)
        ipa = convert_text(
            line, g2p, beam=args.beam,
            segmenter=segmenter if default_seg else None,
            normalize_fn=normalize_fn,
        )
        print(ipa)
        return 0

    # Interactive REPL
    return repl(
        g2p=g2p,
        normalize_fn=normalize_fn,
        segmenter=segmenter,
        default_beam=args.beam,
        default_seg=default_seg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
