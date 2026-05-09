"""Khmer G2P — neural Transformer edition.

Public API (import directly from sub-modules):

    from khmer_g2p.lexicon import load_tsv, Lexicon
    from khmer_g2p.normalizer import normalize, syllabify, tokenize
    from khmer_g2p.segmenter import Segmenter
    from khmer_g2p.neural.infer import NeuralG2P
    from khmer_g2p.neural.phoneme_tokenizer import tokenize as phoneme_tokenize
    from khmer_g2p.hybrid import HybridG2P                # 0.4: lexicon-first wrapper
    from khmer_g2p.neural.finetune import finetune        # 0.4: low-LR resume
    from khmer_g2p.neural.pretrain import pretrain        # 0.5: masked-char pretrain
    from khmer_g2p.neural.ensemble import EnsembleG2P     # 0.5: multi-seed averaging

Typical inference workflows:

    # Pure neural — length-normalized beam search by default
    from khmer_g2p.neural.infer import NeuralG2P
    g2p = NeuralG2P.from_checkpoint("checkpoints/best.pt")
    print(g2p.convert("ខ្មែរ"))                            # greedy
    print(g2p.convert("ខ្មែរ", beam=4))                    # GNMT length-norm
    print(g2p.convert("ខ្មែរ", beam=4, length_penalty=0))  # raw log-prob

    # Hybrid lexicon-first, neural-fallback
    from khmer_g2p.hybrid import HybridG2P
    g2p = HybridG2P.from_paths("checkpoints/best.pt", "data/lexicon.tsv")

    # Multi-seed ensemble (logit averaging) — requires multiple trained checkpoints
    from khmer_g2p.neural.ensemble import EnsembleG2P
    g2p = EnsembleG2P.from_checkpoints([
        "checkpoints_v04/best.pt",
        "checkpoints_v04_ft/best.pt",
    ])
    g2p.convert("ខ្មែរ", beam=4)
"""

__version__ = "0.5.0"
