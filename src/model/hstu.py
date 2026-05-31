import torch
from torch import nn, Tensor
from torch.nn import functional as F

from src.registry import register
from src.model.base_model import BaseModel
from src.datasets.base_dataset import UserHistoryBatch

class HSTUBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_head: int,
        dropout: float
    ) -> None:
        super().__init__()
        assert model_dim % n_head == 0, "model_dim % n_head should be 0"

        self._model_dim = model_dim
        self._n_head = n_head
        self._head_dim = model_dim // n_head

        self._norm = nn.LayerNorm(model_dim)
        self._uqkv_proj = nn.Linear(model_dim, 4 * model_dim)
        self._attn_norm = nn.LayerNorm(model_dim)
        self._out_proj = nn.Linear(model_dim, model_dim)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor):
        B, L, D = x.shape
        device = x.device

        residual = x
        x = self._norm(x)
        u, v, q, k = self._uqkv_proj(x).chunk(4, dim=-1)

        u = F.silu(u)
        q = F.silu(q)
        k = F.silu(k)

        q = q.view(B, L, self._n_head, self._head_dim).transpose(1, 2)
        k = k.view(B, L, self._n_head, self._head_dim).transpose(1, 2)
        v = v.view(B, L, self._n_head, self._head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3)
        scores = scores / (self._head_dim ** 0.5)

        causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        scores = scores.masked_fill(~causal_mask, 0)

        key_padding_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~key_padding_mask, 0)

        attn = F.silu(scores)
        attn = self._dropout(attn)

        hidden = attn @ v
        hidden = hidden.transpose(1, 2).contiguous().view(B, L, D)
        hidden = self._attn_norm(hidden) * u
        hidden = self._out_proj(hidden)

        out = residual + self._dropout(hidden)
        return out


@register("model")
class HSTU(BaseModel):
    def __init__(
        self,
        item_count: int,
        max_history_size: int,
        embedding_dim: int,
        tabular_dim: int = 0,
        attn_n_layers: int = 2,
        attn_n_head: int = 1,
        attn_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._tabular_dim = tabular_dim
        self._model_dim = embedding_dim + tabular_dim

        self._embedding = nn.Embedding(item_count, embedding_dim, padding_idx=0)
        self._pos_embedding = nn.Embedding(max_history_size, embedding_dim)
        self._hstu = nn.ModuleList([
            HSTUBlock(
                self._model_dim,
                attn_n_head,
                attn_dropout
            )
            for _ in range(attn_n_layers)
        ])
        self._final_norm = nn.LayerNorm(self._model_dim)
        self._out_proj: nn.Module = (
            nn.Linear(self._model_dim, embedding_dim) if tabular_dim > 0 else nn.Identity()
        )

    def forward(self, batch: UserHistoryBatch) -> dict[str, Tensor]:
        B, L = batch.history_ids.shape
        device = batch.history_ids.device
        mask = batch.mask.bool()

        item_emb = self._embedding(batch.history_ids)
        positions = torch.arange(L, device=device).expand(B, L)
        x = item_emb + self._pos_embedding(positions)
        if self._tabular_dim > 0:
            x = torch.cat([x, batch.history_features], dim=-1)

        hidden = x * mask.unsqueeze(-1)
        for block in self._hstu:
            hidden = block(hidden, mask)

        hidden = self._final_norm(hidden)
        hidden = self._out_proj(hidden)
        logits = hidden @ self._embedding.weight.T

        out: dict[str, Tensor] = {"logits": logits}
        if batch.target is not None:
            out["targets"] = batch.target.flatten()
            per_pos_loss = F.cross_entropy(
                logits.flatten(0, 1),
                batch.target.flatten(),
                reduction="none",
            )
            m = batch.loss_mask.flatten().float()
            out["loss"] = (per_pos_loss * m).sum() / m.sum().clamp(min=1.0)
        return out
