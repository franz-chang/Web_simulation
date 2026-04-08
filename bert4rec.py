from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class BERT4RecTrainDataset(Dataset):
    def __init__(
        self,
        user_train: dict[int | str, list[int]],
        max_len: int,
        num_items: int,
        mask_prob: float = 0.2,
    ):
        self.user_ids = list(user_train.keys())
        self.user_train = user_train
        self.max_len = max_len
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int):
        uid = self.user_ids[index]
        sequence = self.user_train[uid]
        tokens = sequence[-self.max_len :]

        input_ids = np.zeros(self.max_len, dtype=np.int64)
        labels = np.zeros(self.max_len, dtype=np.int64)

        start = self.max_len - len(tokens)
        for offset, item in enumerate(tokens):
            pos = start + offset
            item = int(item)
            if random.random() < self.mask_prob:
                input_ids[pos] = self.mask_token
                labels[pos] = item
            else:
                input_ids[pos] = item

        if labels.sum() == 0 and len(tokens) > 0:
            pos = self.max_len - 1
            labels[pos] = int(tokens[-1])
            input_ids[pos] = self.mask_token

        return torch.from_numpy(input_ids), torch.from_numpy(labels)


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.mask_token = num_items + 1
        self.vocab_size = num_items + 2  # pad(0) + items + mask

        self.item_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        padding_mask = input_ids.eq(0)

        out = self.transformer(x, src_key_padding_mask=padding_mask)
        out = self.layer_norm(out)
        logits = self.output(out)
        return logits

    def calculate_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=0,
        )
        return loss

    @torch.no_grad()
    def predict_scores(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict next-item by putting [MASK] on last position."""
        logits = self.forward(input_ids)
        last_logits = logits[:, -1, :]
        scores = last_logits[:, : self.num_items + 1]
        scores[:, 0] = -1e9
        return scores
