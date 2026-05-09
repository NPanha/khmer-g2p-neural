"""PyTorch Dataset + collate function for Khmer G2P pairs."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from khmer_g2p.neural.vocab import Vocab


class G2PDataset(Dataset):
    """A dataset of (word, ipa) string pairs, encoded on-the-fly."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: int = 64,
        max_tgt_len: int = 128,
    ) -> None:
        self.pairs = list(pairs)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        word, ipa = self.pairs[idx]
        src_ids = self.src_vocab.encode(word, add_bos=False, add_eos=True)[: self.max_src_len]
        tgt_ids = self.tgt_vocab.encode(ipa, add_bos=True, add_eos=True)[: self.max_tgt_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def make_collate(src_pad_id: int, tgt_pad_id: int):
    """Return a collate_fn that pads source and target to batch maxes."""

    def collate(batch):
        srcs, tgts = zip(*batch)
        src_pad = pad_sequence(srcs, batch_first=True, padding_value=src_pad_id)
        tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=tgt_pad_id)
        # Teacher forcing: input is tgt[:, :-1], target labels are tgt[:, 1:]
        tgt_in = tgt_pad[:, :-1].contiguous()
        tgt_out = tgt_pad[:, 1:].contiguous()
        return src_pad, tgt_in, tgt_out

    return collate


def split_pairs(
    pairs: Sequence[Tuple[str, str]],
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Deterministic train/val/test split. De-duplicates on the source word so
    variant entries for the same word don't straddle splits."""
    # Group entries by word, shuffle unique words, then redistribute.
    by_word: dict = {}
    for w, p in pairs:
        by_word.setdefault(w, []).append((w, p))
    words = list(by_word.keys())
    rng = random.Random(seed)
    rng.shuffle(words)

    n = len(words)
    n_val = max(1, int(n * val_frac)) if n > 20 else 0
    n_test = max(1, int(n * test_frac)) if n > 20 else 0
    n_train = n - n_val - n_test

    train_words = words[:n_train]
    val_words = words[n_train : n_train + n_val]
    test_words = words[n_train + n_val :]

    def flatten(ws):
        out = []
        for w in ws:
            out.extend(by_word[w])
        return out

    return flatten(train_words), flatten(val_words), flatten(test_words)
