from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class GRU4RecTrainDataset(Dataset):
    def __init__(self, user_train: dict[int | str, list[int]], max_len: int):
        self.user_ids = list(user_train.keys())
        self.user_train = user_train
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int):
        uid = self.user_ids[index]
        sequence = self.user_train[uid]

        seq = np.zeros(self.max_len, dtype=np.int64)
        target = np.zeros(self.max_len, dtype=np.int64)

        # Predict next-item at each position: seq[t] -> target[t]
        truncated = sequence[-(self.max_len + 1) :]
        inputs = truncated[:-1]
        labels = truncated[1:]

        if inputs:
            seq[-len(inputs) :] = np.array(inputs, dtype=np.int64)
            target[-len(labels) :] = np.array(labels, dtype=np.int64)

        return torch.from_numpy(seq), torch.from_numpy(target)


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(seq)
        h, _ = self.gru(x)
        h = self.dropout(h)
        logits = self.output(h)
        return logits

    def calculate_loss(self, seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.forward(seq)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=0,
        )
        return loss

    @torch.no_grad()
    def predict_scores(self, seq: torch.Tensor) -> torch.Tensor:
        """Return scores for next item prediction based on last non-padding step."""
        logits = self.forward(seq)
        lengths = torch.clamp(seq.ne(0).sum(dim=1) - 1, min=0)
        last_logits = logits[torch.arange(seq.size(0), device=seq.device), lengths]
        last_logits[:, 0] = -1e9
        return last_logits
