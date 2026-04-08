from __future__ import annotations

import torch
from torch import nn


class BPRMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0.0)
            self.item_embedding.weight[0].fill_(0.0)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u * i).sum(dim=-1)

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_scores = self.score(users, pos_items)
        neg_scores = self.score(users, neg_items)
        ranking_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

        user_reg = self.user_embedding(users).pow(2).sum(dim=1)
        pos_reg = self.item_embedding(pos_items).pow(2).sum(dim=1)
        neg_reg = self.item_embedding(neg_items).pow(2).sum(dim=1)
        reg_loss = (user_reg + pos_reg + neg_reg).mean()
        return ranking_loss, reg_loss

    @torch.no_grad()
    def item_scores_from_history(self, history_items: torch.Tensor) -> torch.Tensor:
        """Infer scores from item history by averaging item embeddings.

        This enables session-style inference for a user-less web demo.
        """
        history_emb = self.item_embedding(history_items)
        profile = history_emb.mean(dim=0, keepdim=True)
        scores = torch.matmul(profile, self.item_embedding.weight.transpose(0, 1))[0]
        scores[0] = -1e9
        return scores
