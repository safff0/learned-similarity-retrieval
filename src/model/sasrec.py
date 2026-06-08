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
        attn_n_head: int = 2,
        attn_dim_feedforward: int = 128,
        attn_dropout: float = 0.1,
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

    def forward(self, batch: UserHistoryBatch) -> dict[str, Tensor]:
        B, L = batch.history_ids.shape
        device = batch.history_ids.device
        item_emb = self._embedding(batch.history_ids)
        positions = torch.arange(L, device=device).expand(B, L)
        x = item_emb + self._pos_embedding(positions)
        if self._tabular_dim > 0:
            x = torch.cat([x, batch.history_features], dim=-1)

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
        hidden = self._out_proj(hidden)
        logits = hidden @ self._embedding.weight.T

        out: dict[str, Tensor] = {"logits": logits}
        if self._use_mol or self._use_similarity_head:
            out["retrieval_queries"] = hidden
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
