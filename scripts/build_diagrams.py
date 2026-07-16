"""Generate clean documentation diagrams into docs/.

Run:
    conda run -n ai python scripts/build_diagrams.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


COLORS = {
    "bg": "#f7f9fc",
    "text": "#14213d",
    "muted": "#516173",
    "line": "#64748b",
    "blue": "#2f80ed",
    "green": "#27ae60",
    "orange": "#f2994a",
    "purple": "#7b61ff",
    "red": "#d64545",
    "teal": "#0f766e",
    "slate": "#475569",
    "white": "#ffffff",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


TITLE = font(34, True)
SUB = font(18)
HEAD = font(21, True)
BODY = font(16)
SMALL = font(14)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1600, 950), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((70, 50), title, fill=COLORS["text"], font=TITLE)
    d.text((72, 94), subtitle, fill=COLORS["muted"], font=SUB)
    return img, d


def save(img: Image.Image, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / name
    img.save(out)
    print(out)


def text_size(d: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = d.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def card(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    color: str,
    fill: str | None = None,
) -> None:
    fill = fill or COLORS["white"]
    x1, y1, x2, _ = xy
    d.rounded_rectangle(xy, radius=14, fill=fill, outline=color, width=2)
    d.text((x1 + 22, y1 + 18), title, fill=color, font=HEAD)
    y = y1 + 58
    for line in lines:
        for part in wrap(line, width=max(24, (x2 - x1) // 12)):
            d.text((x1 + 22, y), part, fill=COLORS["text"], font=BODY)
            y += 25


def solid(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    color: str,
) -> None:
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=16, fill=color)
    tw, th = text_size(d, title, HEAD)
    d.text((x1 + (x2 - x1 - tw) // 2, y1 + 34), title, fill="white", font=HEAD)
    lines = wrap(subtitle, width=max(18, (x2 - x1) // 12))
    y = y1 + 74
    for line in lines:
        lw, _ = text_size(d, line, BODY)
        d.text((x1 + (x2 - x1 - lw) // 2, y), line, fill="#e8eef8", font=BODY)
        y += 24


def arrow(
    d: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = COLORS["line"],
) -> None:
    sx, sy = start
    ex, ey = end
    d.line((sx, sy, ex, ey), fill=color, width=3)
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex >= sx else -1
        pts = [(ex, ey), (ex - 13 * sign, ey - 8), (ex - 13 * sign, ey + 8)]
    else:
        sign = 1 if ey >= sy else -1
        pts = [(ex, ey), (ex - 8, ey - 13 * sign), (ex + 8, ey - 13 * sign)]
    d.polygon(pts, fill=color)


def make_system() -> None:
    img, d = canvas(
        "System Architecture",
        "End-to-end Khmer text to pronunciation with lexicon-first neural fallback",
    )
    y, h, w = 175, 135, 245
    xs = [70, 345, 620, 895, 1170]
    steps = [
        ("Input", "Raw Khmer text or single word", COLORS["blue"]),
        ("Normalize", "NFC, digit cleanup, tokenization", COLORS["green"]),
        ("Segment", "Greedy longest lexicon match", COLORS["green"]),
        ("Hybrid G2P", "Lexicon first, model fallback", COLORS["orange"]),
        ("Output", "Pronunciation sequence for TTS", COLORS["purple"]),
    ]
    for i, (title, sub, color) in enumerate(steps):
        solid(d, (xs[i], y, xs[i] + w, y + h), title, sub, color)
        if i:
            arrow(d, (xs[i - 1] + w, y + h // 2), (xs[i], y + h // 2))

    card(
        d,
        (120, 430, 480, 665),
        "Pronunciation Lexicon",
        ["data/lexicon.tsv", "69,177 entries in saved audit", "Space-separated units: kh, ph, aa, ie"],
        COLORS["green"],
    )
    card(
        d,
        (620, 430, 980, 665),
        "Neural Fallback",
        ["Transformer encoder-decoder", "Greedy or beam search", "Used for unknown words only"],
        COLORS["purple"],
    )
    card(
        d,
        (1120, 430, 1480, 665),
        "Runtime Behavior",
        ["Known words stay stable", "Unknown words still get predictions", "Per-word cache avoids repeated inference"],
        COLORS["teal"],
    )
    arrow(d, (980, 530), (1120, 530), COLORS["line"])
    d.text((120, 735), "Main code paths", fill=COLORS["slate"], font=HEAD)
    d.text(
        (120, 770),
        "pipeline.py orchestrates text processing; hybrid.py chooses lexicon or model; neural/infer.py loads best.pt.",
        fill=COLORS["text"],
        font=BODY,
    )
    d.text(
        (120, 800),
        "scripts/audit_lexicon.py and scripts/clean_lexicon.py support data quality before training.",
        fill=COLORS["text"],
        font=BODY,
    )
    save(img, "system.png")


def make_model() -> None:
    img, d = canvas(
        "Model Architecture",
        "G2PTransformer: Khmer character encoder plus pronunciation decoder",
    )
    card(
        d,
        (80, 185, 390, 780),
        "Source Input",
        ["Khmer word", "Character ids", "src_vocab.json", "EOS appended"],
        COLORS["blue"],
    )
    card(
        d,
        (470, 185, 790, 780),
        "Encoder x6",
        ["Embedding + position", "Self-attention, 8 heads", "Feed-forward hidden size 1536", "LayerNorm and residuals", "Outputs memory"],
        COLORS["purple"],
    )
    card(
        d,
        (870, 185, 1190, 780),
        "Decoder x3",
        ["Target embedding", "Masked self-attention", "Cross-attention over memory", "Output projection", "Tied embeddings enabled"],
        COLORS["purple"],
    )
    card(
        d,
        (1270, 185, 1510, 360),
        "CTC Head",
        ["Connected to encoder memory", "Training-only", "Weight 0.3"],
        COLORS["orange"],
    )
    card(
        d,
        (1270, 415, 1510, 600),
        "Aux Heads",
        ["Optional training heads", "Consonant series labels", "Syllable boundary labels"],
        COLORS["teal"],
    )
    card(
        d,
        (1270, 650, 1510, 820),
        "Decode",
        ["Greedy or beam", "Length penalty 0.6", "max_len from checkpoint"],
        COLORS["blue"],
    )
    arrow(d, (390, 490), (470, 490))
    arrow(d, (790, 490), (870, 490))
    arrow(d, (1190, 735), (1270, 735), COLORS["blue"])
    save(img, "model.png")


def make_training() -> None:
    img, d = canvas(
        "Training Pipeline",
        "From lexicon TSV to best checkpoint and test metrics",
    )
    y, h, w = 175, 130, 210
    xs = [70, 320, 570, 820, 1070, 1320]
    steps = [
        ("Data", "TSV lexicon\n90/5/5 split", COLORS["green"]),
        ("Vocab", "Khmer source\npronunciation target", COLORS["blue"]),
        ("Optional Pretrain", "Masked character\nencoder warm start", COLORS["slate"]),
        ("Train", "CE + CTC\nEMA weights", COLORS["orange"]),
        ("Fine-tune", "Low LR resume\noptional", COLORS["orange"]),
        ("Artifacts", "best.pt\nmetrics JSON", COLORS["purple"]),
    ]
    for i, (title, sub, color) in enumerate(steps):
        solid(d, (xs[i], y, xs[i] + w, y + h), title, sub, color)
        if i:
            arrow(d, (xs[i - 1] + w, y + h // 2), (xs[i], y + h // 2))

    cards = [
        ((90, 455, 430, 710), "Optimization", ["AdamW", "Cosine LR with warmup", "Peak LR 5e-4", "Gradient clip 1.0"], COLORS["orange"]),
        ((485, 455, 825, 710), "Losses", ["Cross entropy", "Label smoothing 0.1", "CTC weight 0.3", "Optional aux losses"], COLORS["red"]),
        ((880, 455, 1220, 710), "Validation", ["Early stopping by val PER", "EMA weights evaluated", "history.json records epochs"], COLORS["teal"]),
        ((1275, 455, 1515, 710), "Current Result", ["PER 3.24%", "WER 15.91%", "n = 3,458 test pairs"], COLORS["purple"]),
    ]
    for xy, title, lines, color in cards:
        card(d, xy, title, lines, color)
    save(img, "training.png")


def make_hybrid() -> None:
    img, d = canvas(
        "Hybrid G2P Decision Logic",
        "Known words use the lexicon; unknown words use the neural model",
    )
    card(d, (80, 220, 390, 380), "Input Word", ["Normalize to NFC", "Use as lookup key"], COLORS["blue"])
    card(d, (80, 570, 390, 760), "Lexicon", ["Canonical pronunciation", "Variants preserved", "hit_rate diagnostics"], COLORS["green"])
    card(d, (575, 325, 895, 555), "Decision", ["Is normalized word in lexicon?", "Yes: return lexicon entry", "No: call neural model"], COLORS["orange"])
    card(d, (1100, 220, 1480, 390), "Lexicon Hit", ["Return stable pronunciation", "last_source = 'lexicon'"], COLORS["green"])
    card(d, (1100, 560, 1480, 760), "Neural Fallback", ["NeuralG2P.convert()", "Beam search optional", "last_source = 'model'"], COLORS["purple"])
    card(d, (575, 680, 895, 830), "Pipeline Cache", ["Memoizes repeated words", "Useful for long TTS metadata"], COLORS["teal"])
    arrow(d, (390, 300), (575, 415))
    arrow(d, (390, 665), (575, 470), COLORS["green"])
    arrow(d, (895, 410), (1100, 305), COLORS["green"])
    arrow(d, (895, 495), (1100, 660), COLORS["purple"])
    arrow(d, (1290, 760), (895, 755), COLORS["teal"])
    save(img, "hybrid.png")


def main() -> None:
    make_system()
    make_model()
    make_training()
    make_hybrid()


if __name__ == "__main__":
    main()
