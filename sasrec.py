import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def negative_sample(positive_set: set[int], num_items: int) -> int:
    sample = random.randint(1, num_items)
    while sample in positive_set:
        sample = random.randint(1, num_items)
    return sample


class SASRecTrainDataset(Dataset):
    def __init__(self, user_train: dict[int, list[int]], num_items: int, max_len: int):
        self.user_ids = list(user_train.keys())
        self.user_train = user_train
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int):
        user_id = self.user_ids[index]
        sequence = self.user_train[user_id]
        sequence_set = set(sequence)

        seq = np.zeros(self.max_len, dtype=np.int64)
        pos = np.zeros(self.max_len, dtype=np.int64)
        neg = np.zeros(self.max_len, dtype=np.int64)

        nxt = sequence[-1]
        fill_idx = self.max_len - 1
        for i in range(len(sequence) - 2, -1, -1):
            seq[fill_idx] = sequence[i]
            pos[fill_idx] = nxt
            if nxt != 0:
                neg[fill_idx] = negative_sample(sequence_set, self.num_items)
            nxt = sequence[i]
            fill_idx -= 1
            if fill_idx < 0:
                break

        return (
            torch.from_numpy(seq),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
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
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seq) + self.position_embedding(positions)
        x = self.dropout(x)

        padding_mask = seq.eq(0)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.layer_norm(out)

    def calculate_loss(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        sequence_output = self.forward(seq)

        pos_emb = self.item_embedding(pos)
        neg_emb = self.item_embedding(neg)
        pos_logits = (sequence_output * pos_emb).sum(dim=-1)
        neg_logits = (sequence_output * neg_emb).sum(dim=-1)

        is_valid = pos.gt(0).float()
        pos_loss = self.bce_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = self.bce_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = (pos_loss + neg_loss) * is_valid
        return loss.sum() / torch.clamp(is_valid.sum(), min=1.0)

    @torch.no_grad()
    def predict_scores(self, seq: torch.Tensor) -> torch.Tensor:
        sequence_output = self.forward(seq)
        lengths = torch.clamp(seq.ne(0).sum(dim=1) - 1, min=0)
        final_output = sequence_output[torch.arange(seq.size(0), device=seq.device), lengths]
        scores = torch.matmul(final_output, self.item_embedding.weight.transpose(0, 1))
        scores[:, 0] = -1e9
        return scores
