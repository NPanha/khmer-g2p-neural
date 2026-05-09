"""Neural (Transformer) G2P for Khmer.

Trains a small seq2seq Transformer on a word→IPA lexicon. Supports
character-level or phoneme-level target tokenization (recommended).

Typical workflow:

    # 1. Train on your TSV
    khmer-g2p-train --data my_lexicon.tsv --out checkpoints/ --tgt-tokenizer phoneme

    # 2. Predict
    khmer-g2p-predict --ckpt checkpoints/best.pt "ខ្មែរ"

    # 3. Use in Python
    from khmer_g2p.neural import NeuralG2P
    model = NeuralG2P.from_checkpoint("checkpoints/best.pt")
    print(model.convert("ខ្មែរ"))
"""

# Lazy imports — torch is an optional dependency.
__all__ = ["NeuralG2P", "train", "Vocab"]


def __getattr__(name: str):
    if name == "NeuralG2P":
        from khmer_g2p.neural.infer import NeuralG2P
        return NeuralG2P
    if name == "train":
        from khmer_g2p.neural.train import train
        return train
    if name == "Vocab":
        from khmer_g2p.neural.vocab import Vocab
        return Vocab
    raise AttributeError(f"module 'khmer_g2p.neural' has no attribute {name!r}")
