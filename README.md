# khmer-g2p-neural

Neural (Transformer) grapheme-to-phoneme conversion for Khmer.

A character-level encoder-decoder trained on a Khmer word → phoneme lexicon.
The current lexicon uses a Khmer word plus a space-separated pronunciation
string. It is IPA-based, but uses project-specific ASCII-friendly units for
some sounds: aspiration is written as digraphs (`ph`, `th`, `kh`, `ch`) and
long vowels are written by doubling (`aa`, `ii`, `oo`, `ɑɑ`, `əə`, ...).
No FST dependency — pure PyTorch.

## Pipeline overview

```
Install → Prepare data → (Pretrain) → Train → Fine-tune → Predict / Evaluate
```

Steps in parentheses are optional but recommended when you have extra data.

## 1. Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- tqdm ≥ 4.60

GPU is optional but strongly recommended for training.

## 2. Install

Clone the repo and install in editable mode with all extras:

```bash
git clone https://github.com/NPanha/khmer-g2p-neural.git
cd khmer-g2p-neural
pip install -e ".[notebook,dev]"
```

`[notebook]` adds matplotlib, seaborn, pandas, numpy, and jupyter.
`[dev]` adds pytest.

Verify the install:

```bash
python -c "import khmer_g2p; print(khmer_g2p.__version__)"
khmer-g2p-train --help
```

## 3. Prepare data

### Lexicon format

The lexicon is a UTF-8 two-column TSV: Khmer word and pronunciation, one pair
per line. A header row such as `word<TAB>ipa` is allowed and will be skipped
automatically.

Each pronunciation is written as space-separated phoneme units. Use `.` as a
syllable boundary inside the pronunciation when needed.

```
ខ្មែរ	kh m ae r
ភាសា	ph ie s aa
កម្ពុជា	k ɑm p u c ie
```

Expected notation in this project:

| Type | Examples |
|------|----------|
| Aspirated consonants | `ph`, `th`, `kh`, `ch` |
| Long vowels | `aa`, `ii`, `oo`, `ee`, `uu`, `ɑɑ`, `əə`, `ɔɔ`, `ɛɛ`, `ɨɨ` |
| Vowel clusters / diphthongs | `ie`, `ea`, `oa`, `ae`, `uə`, `iə`, `ao`, `aə`, `ɨə`, `ɛə` |
| Separators | space between phoneme units, `.` between syllables |

The training commands expect the full lexicon at `data/lexicon.tsv`. This file
is data, not model code; make sure it exists locally before training.

The latest saved audit for this project reported:

- 69,177 rows loaded
- 69,168 unique words
- 32 distinct target tokens after tokenizer processing
- 0 duplicate rows
- 0 unknown phonemes
- 0 whitespace issues

### Audit the lexicon

Before training, check for duplicates, conflicting variants, rare or unknown
phonemes, and whitespace drift:

```bash
python scripts/audit_lexicon.py data/lexicon.tsv
# save a full report:
python scripts/audit_lexicon.py data/lexicon.tsv --top 40 --out audit.txt
```

### Clean the lexicon

Fix the issues the audit finds — trailing punctuation, `g`/`ɡ` confusion,
empty pronunciation fields, standalone vowel/punctuation entries, and duplicate
rows:

```bash
# dry-run first to preview changes
python scripts/clean_lexicon.py data/lexicon.tsv --dry-run

# apply and write cleaned file
python scripts/clean_lexicon.py data/lexicon.tsv \
    --out data/lexicon.clean.tsv \
    --drop-empty-ipa --drop-vowel-punct --dedupe
```

Use `data/lexicon.clean.tsv` for training if you ran the cleaner.

## 4. (Optional) Pretrain the encoder

If you have a raw Khmer text corpus, masked-character pretraining of the
encoder typically improves PER by 1–3 points on low-resource lexicons.
The architecture flags must match what you plan to use in supervised training.

```bash
khmer-g2p-pretrain \
    --corpus path/to/khmer_corpus.txt \
    --lexicon data/lexicon.tsv \
    --out checkpoints_pretrain
```

This writes `checkpoints_pretrain/pretrain.pt`, which you pass to `--pretrained-encoder`
in the next step.

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | required | Plain-text Khmer file, one document per line |
| `--lexicon` | — | Adds lexicon characters to the vocab for clean transfer |
| `--out` | `checkpoints_pretrain` | Output directory |
| `--mlm-prob` | 0.15 | Fraction of characters masked |
| `--epochs` | 10 | Training epochs |
| `--d-model` | 384 | Must match supervised training |
| `--enc-layers` | 6 | Must match supervised training |

## 5. Train

### Minimal command

```bash
khmer-g2p-train --data data/lexicon.tsv --out checkpoints_v04
```

### All defaults spelled out

```bash
khmer-g2p-train --data data/lexicon.tsv --out checkpoints_v04 \
    --d-model 384 --nhead 8 --enc-layers 6 --dec-layers 3 \
    --ffn 1536 --dropout 0.3 \
    --tgt-tokenizer phoneme \
    --ctc --ctc-weight 0.3 \
    --epochs 80 --patience 12 \
    --lr 5e-4 --warmup 1000 --weight-decay 0.01
```

### With a pretrained encoder

```bash
khmer-g2p-train --data data/lexicon.tsv --out checkpoints_v04 \
    --pretrained-encoder checkpoints_pretrain/pretrain.pt
```

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | TSV lexicon path |
| `--out` | `checkpoints` | Output directory for checkpoints and vocabs |
| `--d-model` | 384 | Embedding / hidden size |
| `--nhead` | 8 | Attention heads |
| `--enc-layers` | 6 | Encoder layers |
| `--dec-layers` | 3 | Decoder layers |
| `--ffn` | 1536 | Feed-forward inner size |
| `--dropout` | 0.3 | Dropout probability |
| `--tgt-tokenizer` | `phoneme` | Target tokenizer. With this lexicon, space-separated units like `kh`, `aa`, and `ie` are preserved in the raw pronunciation string while training still uses the project tokenizer/vocab saved in the checkpoint. |
| `--ctc` / `--no-ctc` | on | CTC auxiliary loss on encoder |
| `--ctc-weight` | 0.3 | Weight of CTC in total loss |
| `--epochs` | 80 | Max epochs |
| `--patience` | 12 | Early-stop patience (epochs without val PER improvement) |
| `--lr` | 5e-4 | Peak learning rate |
| `--warmup` | 1000 | Linear warmup steps |
| `--lr-schedule` | `cosine` | `cosine` or `noam` |
| `--amp` | `auto` | Mixed precision: `auto`, `fp16`, `bf16`, `off` |
| `--ema-decay` | 0.999 | EMA of weights; 0 disables |
| `--series-weight` | 0.0 | Aux loss on consonant-series head (0.1–0.3 recommended if used) |
| `--syllable-weight` | 0.0 | Aux loss on syllable-boundary head (0.1–0.3 recommended if used) |
| `--val-frac` | 0.05 | Fraction of data for validation |
| `--test-frac` | 0.05 | Fraction of data for test |
| `--seed` | 42 | Random seed |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

### CTC auxiliary loss

The encoder carries an optional CTC head. Total training loss is:

```
loss = CE(decoder) + ctc_weight × CTC(encoder)
```

This regularizes the encoder toward a monotonic phoneme alignment and is
typically worth 1–2 PER points on low-resource G2P. The CTC head is
training-only — inference is unchanged.

### Output files

After training the `--out` directory contains:

```
checkpoints_v04/
├── best.pt           # best checkpoint by val PER  ← use this for inference
├── last.pt           # final epoch checkpoint
├── src_vocab.json    # source character vocabulary
├── tgt_vocab.json    # target pronunciation vocabulary
├── history.json      # per-epoch metrics (loss, PER, WER, LR)
└── test_metrics.json # final test-set PER / WER
```

### Reproduce the old small-model baseline

```bash
khmer-g2p-train --data data/lexicon.tsv --out checkpoints_baseline \
    --no-ctc --d-model 192 --nhead 4 --enc-layers 3 --dec-layers 3 \
    --ffn 512 --dropout 0.1 --epochs 30 --tgt-tokenizer char \
    --warmup 500 --weight-decay 1e-5 --patience 5
```

## 6. Fine-tune

Fine-tuning resumes from a trained checkpoint with a flat low LR, reduced
dropout, and CTC disabled — useful after receiving additional annotated data
or for domain adaptation:

```bash
khmer-g2p-finetune \
    --ckpt checkpoints_v04/best.pt \
    --data data/lexicon.tsv \
    --out checkpoints_finetune
```

Key fine-tuning flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt` | required | Checkpoint to resume from |
| `--data` | required | TSV lexicon (same split as training) |
| `--out` | `checkpoints_finetune` | New output directory |
| `--lr` | 5e-5 | Flat learning rate (no warmup/decay) |
| `--dropout` | 0.1 | Reduced dropout |
| `--epochs` | 20 | Max fine-tune epochs |
| `--patience` | 6 | Early-stop patience |
| `--ctc-weight` | 0.0 | CTC off by default; set > 0 to keep it |

## 7. Predict

### CLI — single word

```bash
khmer-g2p-predict --ckpt checkpoints_v04/best.pt "ខ្មែរ"
```

### CLI — batch (stdin)

```bash
echo "ខ្មែរ" | khmer-g2p-predict --ckpt checkpoints_v04/best.pt --batch
```

### CLI — batch (file, one word per line)

```bash
khmer-g2p-predict --ckpt checkpoints_v04/best.pt -f words.txt
```

### CLI — beam search

```bash
khmer-g2p-predict --ckpt checkpoints_v04/best.pt --beam 4 "ខ្មែរ"
```

### CLI — hybrid (lexicon-first, neural fallback)

```bash
khmer-g2p-predict --ckpt checkpoints_v04/best.pt \
    --lexicon data/lexicon.tsv --beam 4 "ខ្មែរ"
```

### CLI — ensemble

```bash
khmer-g2p-predict \
    --ensemble checkpoints_v04/best.pt checkpoints_v04b/best.pt \
    --beam 4 "ខ្មែរ"
```

### Python API

```python
from khmer_g2p.neural.infer import NeuralG2P

g2p = NeuralG2P.from_checkpoint("checkpoints_v04/best.pt")
print(g2p.convert("ខ្មែរ"))           # greedy
print(g2p.convert("ខ្មែរ", beam=4))   # beam search

# Lexicon-first with neural fallback
from khmer_g2p.hybrid import HybridG2P
from khmer_g2p.lexicon import Lexicon, load_tsv

g2p = HybridG2P(
    NeuralG2P.from_checkpoint("checkpoints_v04/best.pt"),
    Lexicon(load_tsv("data/lexicon.tsv")),
)
print(g2p.convert("ខ្មែរ"))
print(g2p.last_source)   # 'lexicon' or 'model'
```

## 8. Evaluate (notebook)

`examples/inference_evaluation.ipynb` walks through the full evaluation workflow after
training: load checkpoint, single-word and batch inference, beam search
comparison, PER/WER on the test split, training history plot, lexicon
hit-rate stats, and error analysis of worst predictions.

```bash
jupyter notebook examples/inference_evaluation.ipynb
```

The notebook anchors itself to the repo root automatically — open it from
any working directory.

## 9. Tests

```bash
pytest tests/ -v
```

---

## Layout

```
src/khmer_g2p/
├── phonemes.py              # Khmer phonological inventory (consonant series, vowels)
├── lexicon.py               # TSV loader + dict lookup
├── normalizer.py            # NFC normalization, syllabification, tokenization
├── segmenter.py             # greedy longest-match word segmenter
├── hybrid.py                # lexicon-first G2P with neural fallback
└── neural/
    ├── phoneme_tokenizer.py # target pronunciation tokenizer + inventory checks
    ├── vocab.py             # char / phoneme vocab
    ├── dataset.py           # PyTorch Dataset + collate
    ├── model.py             # Transformer encoder-decoder + CTC/aux heads
    ├── aux_labels.py        # per-position series + syllable-boundary labels
    ├── train.py             # training loop (tqdm, history.json)
    ├── pretrain.py          # masked-character pretraining
    ├── finetune.py          # low-LR fine-tuning resume
    ├── ema.py               # exponential moving average of weights
    ├── ensemble.py          # multi-checkpoint logit averaging
    ├── infer.py             # NeuralG2P (greedy + beam)
    ├── metrics.py           # PER, WER, BLEU, confusion, P/R/F1, buckets
    └── cli.py               # train / predict / finetune / pretrain entry points

scripts/
├── audit_lexicon.py         # data-quality report
├── clean_lexicon.py         # remove trailing junk, normalize g↔ɡ
└── live_test.py             # interactive spot-check against a running model

examples/
├── inference_evaluation.ipynb      # checkpoint inference + evaluation workflow
├── pipeline_demo.ipynb             # sentence-level pipeline demo
└── colab_training_exploration.ipynb # Colab-oriented training/exploration notes

tests/                        # pytest suite
```
