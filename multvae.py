import torch
from torch import nn


class MultiVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        p_dims: list[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if p_dims is None:
            p_dims = [256, 128]
        self.input_dim = input_dim
        self.p_dims = p_dims + [input_dim]
        self.q_dims = [input_dim] + p_dims[::-1]
        self.dropout = nn.Dropout(dropout)

        q_layers = []
        for i in range(len(self.q_dims) - 1):
            q_layers.append(nn.Linear(self.q_dims[i], self.q_dims[i + 1]))
        self.q_layers = nn.ModuleList(q_layers[:-1])
        latent_dim = self.q_dims[-1]
        self.q_mu = nn.Linear(self.q_dims[-2], latent_dim)
        self.q_logvar = nn.Linear(self.q_dims[-2], latent_dim)

        p_layers = []
        for i in range(len(self.p_dims) - 1):
            p_layers.append(nn.Linear(self.p_dims[i], self.p_dims[i + 1]))
        self.p_layers = nn.ModuleList(p_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(x)
        for layer in self.q_layers:
            h = torch.tanh(layer(h))
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, self.training)
        logits = self.decode(z)
        return logits, mu, logvar

    @torch.no_grad()
    def recommend_logits(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encode(x)
        return self.decode(mu)

    @staticmethod
    def loss_fn(
        logits: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        anneal: float,
    ) -> torch.Tensor:
        log_softmax = torch.log_softmax(logits, dim=1)
        recon_loss = -(log_softmax * x).sum(dim=1).mean()
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + anneal * kl
