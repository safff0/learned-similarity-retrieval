import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from src.registry import register
from src.model.base_model import BaseModel
from src.model.mol_module import (
    MoLHead,
    resolve_mol_head_params,
    resolve_mol_runtime_params,
)
from src.model.similarity_heads import (
    build_similarity_head,
    resolve_similarity_runtime_config,
)
from src.datasets.base_dataset import UserHistoryBatch


def _truncated_normal(x: Tensor, mean: float = 0.0, std: float = 0.02) -> Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.mul_(std).add_(mean)
        return x


@register("model")
class SASRec(BaseModel):
    def __init__(
        self,
        item_count: int,
        max_history_size: int,
        embedding_dim: int,
        tabular_dim: int = 0,
        use_mol: bool = False,
        mol: dict | None = None,
        mol_runtime: dict | None = None,
        use_similarity_head: bool = False,
        similarity_head: dict | None = None,
        similarity_runtime: dict | None = None,
        dot_loss_weight: float = 1.0,
        attn_n_layers: int = 2,
        attn_n_head: int = 1,
        attn_dim_feedforward: int = 50,
        attn_dropout: float = 0.1,
        input_dropout: float = 0.2,
        user_embedding_norm: str = "layer_norm",
    ) -> None:
        super().__init__()
        mol_cfg = resolve_mol_head_params(mol)
        mol_runtime_cfg = resolve_mol_runtime_params(mol_runtime)
        self._embedding_dim = embedding_dim
        self._tabular_dim = tabular_dim
        self._model_dim = embedding_dim + tabular_dim
        self._use_mol = use_mol
        self._use_similarity_head = use_similarity_head
        self._dot_loss_weight = dot_loss_weight
        self._mol_loss_weight = mol_runtime_cfg["loss_weight"]
        self._mol_mi_loss_weight = (
            0.0
            if mol_runtime_cfg["mi_loss_weight"] is None
            else mol_runtime_cfg["mi_loss_weight"]
        )
        self._mol_uid_embedding_l2_weight = mol_runtime_cfg["uid_embedding_l2_weight"]
        runtime_cfg = resolve_similarity_runtime_config(similarity_runtime)
        self._similarity_loss_weight = runtime_cfg["loss_weight"]
        self._similarity_mi_loss_weight = runtime_cfg["mi_loss_weight"] or 0.0
        self._similarity_uid_l2_weight = runtime_cfg["uid_embedding_l2_weight"]
        self._user_embedding_norm = user_embedding_norm
        self._input_dropout = nn.Dropout(input_dropout)

        self._embedding = nn.Embedding(item_count, embedding_dim, padding_idx=0)
        self._pos_embedding = nn.Embedding(max_history_size, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._model_dim,
            nhead=attn_n_head,
            dim_feedforward=attn_dim_feedforward,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=attn_n_layers)
        self._final_norm = nn.LayerNorm(self._model_dim)
        self._out_proj: nn.Module = (
            nn.Linear(self._model_dim, embedding_dim) if tabular_dim > 0 else nn.Identity()
        )
        self._mol_head: MoLHead | None = None
        self._similarity_head_name: str | None = None
        self._similarity_head = None
        if use_mol:
            self._mol_head = MoLHead(
                item_embedding=self._embedding,
                embedding_dim=embedding_dim,
                item_count=item_count,
                **mol_cfg,
            )
        elif use_similarity_head:
            self._similarity_head_name, self._similarity_head = build_similarity_head(
                similarity_head,
                item_embedding=self._embedding,
                embedding_dim=embedding_dim,
                item_count=item_count,
            )
        self.reset_params()

    def reset_params(self) -> None:
        _truncated_normal(self._embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self._embedding.weight[0].zero_()
        _truncated_normal(
            self._pos_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self._embedding_dim),
        )

    def forward(self, batch: UserHistoryBatch) -> dict[str, Tensor]:
        B, L = batch.history_ids.shape
        device = batch.history_ids.device
        item_emb = self._embedding(batch.history_ids)
        positions = torch.arange(L, device=device).expand(B, L)
        x = item_emb * math.sqrt(self._embedding_dim) + self._pos_embedding(positions)
        if self._tabular_dim > 0:
            x = torch.cat([x, batch.history_features], dim=-1)
        x = self._input_dropout(x)

        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1,
        )
        key_padding_mask = ~batch.mask
        hidden = self._encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        hidden = self._final_norm(hidden)
        hidden = self._out_proj(hidden)
        if self._user_embedding_norm == "layer_norm":
            hidden = F.layer_norm(hidden, normalized_shape=(hidden.size(-1),))
        elif self._user_embedding_norm == "l2":
            hidden = F.normalize(hidden, p=2, dim=-1)
        elif self._user_embedding_norm != "none":
            raise ValueError(f"Unknown user_embedding_norm: {self._user_embedding_norm}")
        logits = hidden @ self._embedding.weight.T

        out: dict[str, Tensor] = {"logits": logits, "retrieval_queries": hidden}
        if not self.training:
            last_history_idx = batch.mask.sum(dim=1).long() - 1
            out["next_retrieval_queries"] = hidden[
                torch.arange(hidden.size(0), device=hidden.device),
                last_history_idx,
            ]
        if batch.target is not None:
            out["targets"] = batch.target.flatten()
            total_loss = hidden.new_zeros(())
            if self._dot_loss_weight > 0.0:
                per_pos_loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    batch.target.flatten(),
                    reduction="none",
                )
                m = batch.loss_mask.flatten().float()
                dot_loss = (per_pos_loss * m).sum() / m.sum().clamp(min=1.0)
                out["dot_loss"] = dot_loss
                total_loss = total_loss + self._dot_loss_weight * dot_loss
            if self._use_mol:
                mol_loss, mol_aux = self.compute_mol_training_loss(hidden, batch)
                out.update(mol_aux)
                if mol_loss is not None:
                    out["mol_loss"] = mol_loss
                    total_loss = total_loss + self._mol_loss_weight * mol_loss
                mi_loss = mol_aux.get("mi_loss")
                if self._mol_mi_loss_weight > 0.0 and mi_loss is not None:
                    total_loss = total_loss + self._mol_mi_loss_weight * mi_loss
                uid_l2 = mol_aux.get("uid_embedding_l2_norm")
                if self._mol_uid_embedding_l2_weight > 0.0 and uid_l2 is not None:
                    total_loss = total_loss + self._mol_uid_embedding_l2_weight * uid_l2
            if self._use_similarity_head:
                similarity_loss, similarity_aux = self.compute_similarity_training_loss(hidden, batch)
                out.update(similarity_aux)
                if similarity_loss is not None:
                    out["similarity_loss"] = similarity_loss
                    total_loss = total_loss + self._similarity_loss_weight * similarity_loss
                mi_loss = similarity_aux.get("mi_loss")
                if self._similarity_mi_loss_weight > 0.0 and mi_loss is not None:
                    total_loss = total_loss + self._similarity_mi_loss_weight * mi_loss
                uid_l2 = similarity_aux.get("uid_embedding_l2_norm")
                if self._similarity_uid_l2_weight > 0.0 and uid_l2 is not None:
                    total_loss = total_loss + self._similarity_uid_l2_weight * uid_l2
            out["loss"] = total_loss
        return out
