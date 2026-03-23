import torch
from torch import nn


def build_norm_adj(
    num_users: int,
    num_items: int,
    user_pos_items: dict[int, set[int]],
) -> torch.Tensor:
    rows = []
    cols = []
    for user_idx, items in user_pos_items.items():
        for item_idx in items:
            item_node = num_users + item_idx
            rows.append(user_idx)
            cols.append(item_node)
            rows.append(item_node)
            cols.append(user_idx)

    num_nodes = num_users + num_items + 1
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, adj.indices()[0], adj.values())
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    norm_values = (
        deg_inv_sqrt[adj.indices()[0]]
        * adj.values()
        * deg_inv_sqrt[adj.indices()[1]]
    )
    norm_adj = torch.sparse_coo_tensor(
        adj.indices(),
        norm_values,
        adj.size(),
    ).coalesce()
    return norm_adj


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        embedding_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0.0)
            self.item_embedding.weight[0].fill_(0.0)

        self.register_buffer("norm_adj", norm_adj.coalesce())

    def computer(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight[1:]],
            dim=0,
        )
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            embs.append(x)
        all_out = torch.stack(embs, dim=1).mean(dim=1)

        user_out = all_out[: self.num_users + 1]
        item_out = torch.zeros(
            (self.num_items + 1, all_out.size(1)),
            dtype=all_out.dtype,
            device=all_out.device,
        )
        item_out[1:] = all_out[self.num_users + 1 :]
        return user_out, item_out

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_out, item_out = self.computer()
        u_emb = user_out[users]
        pos_emb = item_out[pos_items]
        neg_emb = item_out[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

        u_ego = self.user_embedding(users)
        p_ego = self.item_embedding(pos_items)
        n_ego = self.item_embedding(neg_items)
        reg = (
            u_ego.pow(2).sum(dim=1)
            + p_ego.pow(2).sum(dim=1)
            + n_ego.pow(2).sum(dim=1)
        ).mean()
        return loss, reg
