import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.datasets.base_dataset import UserHistoryBatch
from src.model.base_model import BaseModel
from src.model.mol_module import (
    MoLHead,
    resolve_mol_head_params,
    resolve_mol_runtime_params,
)
from src.registry import register


def _truncated_normal(x: Tensor, mean: float = 0.0, std: float = 0.02) -> Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.mul_(std).add_(mean)
        return x


class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int = 128,
    ) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        self._num_buckets = num_buckets
        self._ts_w = nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0.0, std=0.02))
        self._pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))

    def forward(self, all_timestamps: Tensor) -> Tensor:
        batch_size = all_timestamps.size(0)
        seq_len = all_timestamps.size(1)
        if seq_len > self._max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum {self._max_seq_len}"
            )
        pos = F.pad(self._pos_w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        pos = pos[..., :-seq_len].reshape(1, seq_len, 3 * seq_len - 2)
        radius = (2 * seq_len - 1) // 2

        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, seq_len - 1 : seq_len]],
            dim=1,
        )
        deltas = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        bucketed = torch.clamp(
            (torch.log(torch.abs(deltas).clamp(min=1)) / 0.301).long(),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = pos[:, :, radius:-radius]
        rel_ts_bias = torch.index_select(self._ts_w, dim=0, index=bucketed.view(-1)).view(
            batch_size, seq_len, seq_len
        )
        return rel_pos_bias + rel_ts_bias



class HSTUBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_head: int,
        dropout: float,
        attn_dropout: float,
        attention_dim: int | None = None,
        linear_hidden_dim: int | None = None,
        linear_activation: str = "silu",
        eps: float = 1e-6,
        relative_attention_bias_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        attention_dim = attention_dim or (model_dim // n_head)
        linear_hidden_dim = linear_hidden_dim or (model_dim // n_head)

        self._model_dim = model_dim
        self._n_head = n_head
        self._attention_dim = attention_dim
        self._linear_hidden_dim = linear_hidden_dim
        self._linear_activation = linear_activation
        self._eps = eps
        self._relative_attention_bias_module = relative_attention_bias_module

        self._uvqk = nn.Parameter(
            torch.empty(
                model_dim,
                2 * linear_hidden_dim * n_head + 2 * attention_dim * n_head,
            ).normal_(mean=0.0, std=0.02)
        )
        self._out_proj = nn.Linear(linear_hidden_dim * n_head, model_dim)
        nn.init.xavier_uniform_(self._out_proj.weight)
        if self._out_proj.bias is not None:
            nn.init.zeros_(self._out_proj.bias)
        self._dropout = nn.Dropout(dropout)
        self._attn_dropout_ratio = attn_dropout

    def forward(self, x: Tensor, mask: Tensor, timestamps: Tensor | None = None) -> Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device

        residual = x
        x = F.layer_norm(x, normalized_shape=[self._model_dim], eps=self._eps)
        uvqk = torch.matmul(x, self._uvqk)
        if self._linear_activation == "silu":
            uvqk = F.silu(uvqk)
        elif self._linear_activation != "none":
            raise ValueError(f"Unsupported linear activation: {self._linear_activation}")

        u, v, q, k = torch.split(
            uvqk,
            [
                self._linear_hidden_dim * self._n_head,
                self._linear_hidden_dim * self._n_head,
                self._attention_dim * self._n_head,
                self._attention_dim * self._n_head,
            ],
            dim=-1,
        )

        q = q.view(batch_size, seq_len, self._n_head, self._attention_dim)
        k = k.view(batch_size, seq_len, self._n_head, self._attention_dim)
        v = v.view(batch_size, seq_len, self._n_head, self._linear_hidden_dim)

        scores = torch.einsum("bnhd,bmhd->bhnm", q, k)
        scores = F.silu(scores) / seq_len
        if self._relative_attention_bias_module is not None and timestamps is not None:
            scores = scores + self._relative_attention_bias_module(timestamps).unsqueeze(1)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        invalid_attn_mask = (causal_mask.unsqueeze(0) & mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1)
        attn = scores * invalid_attn_mask

        hidden = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        hidden = hidden.contiguous().view(
            batch_size,
            seq_len,
            self._linear_hidden_dim * self._n_head,
        )
        hidden = F.layer_norm(
            hidden,
            normalized_shape=[self._linear_hidden_dim * self._n_head],
            eps=self._eps,
        )
        hidden = u * hidden
        hidden = self._out_proj(
            F.dropout(hidden, p=self._dropout.p, training=self.training)
        )

        return residual + hidden


@register("model")
class HSTU(BaseModel):
    def __init__(
        self,
        item_count: int,
        max_history_size: int,
        embedding_dim: int,
        tabular_dim: int = 0,
        rating_embedding_dim: int = 0,
        num_ratings: int = 6,
        attn_n_layers: int = 2,
        attn_n_head: int = 1,
        attn_dropout: float = 0.2,
        attn_linear_dropout: float | None = None,
        attn_attention_dim: int | None = None,
        attn_linear_dim: int | None = None,
        attn_linear_activation: str = "silu",
        input_dropout: float = 0.2,
        output_postproc: str = "l2",
        enable_relative_attention_bias: bool = True,
        use_mol: bool = False,
        mol: dict | None = None,
        mol_runtime: dict | None = None,
        dot_loss_weight: float = 1.0,
        dot_sampled_softmax: bool = False,
        dot_num_train_negatives: int = 128,
        dot_softmax_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        mol_cfg = resolve_mol_head_params(mol)
        mol_runtime_cfg = resolve_mol_runtime_params(mol_runtime)
        self._embedding_dim = embedding_dim
        self._tabular_dim = tabular_dim
        self._rating_embedding_dim = rating_embedding_dim
        self._model_dim = embedding_dim + tabular_dim + rating_embedding_dim
        self._use_mol = use_mol
        self._mol_mi_loss_weight = (
            0.0
            if mol_runtime_cfg["mi_loss_weight"] is None
            else mol_runtime_cfg["mi_loss_weight"]
        )
        self._mol_uid_embedding_l2_weight = mol_runtime_cfg["uid_embedding_l2_weight"]
        self._mol_query_chunk_size = mol_cfg["query_chunk_size"]
        self._mol_loss_weight = mol_runtime_cfg["loss_weight"]
        self._dot_loss_weight = dot_loss_weight
        self._dot_sampled_softmax = dot_sampled_softmax
        self._dot_num_train_negatives = dot_num_train_negatives
        self._dot_softmax_temperature = dot_softmax_temperature
        self._mol_eps = mol_cfg["eps"]
        self._mol_uid_embedding_hash_sizes = mol_cfg["uid_embedding_hash_sizes"] or []
        self._input_dropout = nn.Dropout(input_dropout)
        self._output_postproc = output_postproc
        self._enable_relative_attention_bias = enable_relative_attention_bias
        self.register_buffer(
            "_catalog_item_ids",
            torch.arange(1, item_count, dtype=torch.long),
            persistent=False,
        )

        self._embedding = nn.Embedding(item_count, embedding_dim, padding_idx=0)
        self._rating_embedding: nn.Module = (
            nn.Embedding(num_ratings, rating_embedding_dim, padding_idx=0)
            if rating_embedding_dim > 0
            else nn.Identity()
        )
        positional_dim = embedding_dim + rating_embedding_dim
        self._pos_embedding = nn.Embedding(max_history_size + 1, positional_dim)
        linear_dropout = attn_dropout if attn_linear_dropout is None else attn_linear_dropout
        self._hstu = nn.ModuleList(
            [
                HSTUBlock(
                    self._model_dim,
                    attn_n_head,
                    linear_dropout,
                    attn_dropout,
                    attention_dim=attn_attention_dim,
                    linear_hidden_dim=attn_linear_dim,
                    linear_activation=attn_linear_activation,
                    eps=self._mol_eps,
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_history_size + 1
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                )
                for _ in range(attn_n_layers)
            ]
        )
        self._final_norm = nn.LayerNorm(self._model_dim)
        self._out_proj: nn.Module = (
            nn.Linear(self._model_dim, embedding_dim)
            if self._model_dim != embedding_dim
            else nn.Identity()
        )
        self._mol_head: MoLHead | None = None
        if self._use_mol:
            self._mol_head = MoLHead(
                item_embedding=self._embedding,
                embedding_dim=embedding_dim,
                item_count=item_count,
                **mol_cfg,
            )
        self.reset_params()

    def reset_params(self) -> None:
        _truncated_normal(self._embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self._embedding.weight[0].zero_()
        if isinstance(self._rating_embedding, nn.Embedding):
            _truncated_normal(
                self._rating_embedding.weight,
                mean=0.0,
                std=1.0 / math.sqrt(self._embedding_dim + self._rating_embedding_dim),
            )
            with torch.no_grad():
                self._rating_embedding.weight[0].zero_()
        _truncated_normal(
            self._pos_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self._pos_embedding.embedding_dim),
        )

    def _compute_logits(self, hidden: Tensor) -> Tensor:
        item_embeddings = self._embedding.weight
        if self._output_postproc == "l2":
            item_embeddings = item_embeddings / torch.clamp(
                torch.linalg.norm(item_embeddings, ord=2, dim=-1, keepdim=True),
                min=self._mol_eps,
            )
        return hidden @ item_embeddings.T

    def _compute_dot_loss(self, logits: Tensor, batch: UserHistoryBatch) -> Tensor:
        per_pos_loss = F.cross_entropy(
            logits.flatten(0, 1),
            batch.target.flatten(),
            reduction="none",
        )
        mask = batch.loss_mask.flatten().float()
        return (per_pos_loss * mask).sum() / mask.sum().clamp(min=1.0)

    def _compute_sampled_dot_training_loss(
        self,
        hidden: Tensor,
        batch: UserHistoryBatch,
    ) -> Tensor | None:
        valid = batch.loss_mask.bool()
        positions = valid.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return None

        batch_indices = positions[:, 0]
        time_indices = positions[:, 1]
        query_embeddings = hidden[batch_indices, time_indices]
        target_items = batch.target[batch_indices, time_indices].long()
        if target_items.numel() == 0:
            return None

        total_loss = hidden.new_zeros(())
        num_examples = target_items.numel()
        query_chunk_size = max(1, self._mol_query_chunk_size)
        num_negatives = min(self._dot_num_train_negatives, self._catalog_item_ids.numel())

        for start in range(0, num_examples, query_chunk_size):
            end = min(start + query_chunk_size, num_examples)
            query_chunk = query_embeddings[start:end]
            target_chunk = target_items[start:end]

            positive_embeddings = self._embedding(target_chunk)
            if self._output_postproc == "l2":
                positive_embeddings = positive_embeddings / torch.clamp(
                    torch.linalg.norm(positive_embeddings, ord=2, dim=-1, keepdim=True),
                    min=self._mol_eps,
                )
            positive_logits = (
                (query_chunk * positive_embeddings).sum(dim=-1, keepdim=True)
                / self._dot_softmax_temperature
            )

            sampled_ids, sampled_negative_embeddings = self._sample_local_negatives(
                positive_ids=target_chunk,
                num_to_sample=num_negatives,
            )
            if self._output_postproc == "l2":
                sampled_negative_embeddings = sampled_negative_embeddings / torch.clamp(
                    torch.linalg.norm(
                        sampled_negative_embeddings, ord=2, dim=-1, keepdim=True
                    ),
                    min=self._mol_eps,
                )
            sampled_negatives_logits = (
                torch.einsum("bd,bnd->bn", query_chunk, sampled_negative_embeddings)
                / self._dot_softmax_temperature
            )
            sampled_negatives_logits = torch.where(
                target_chunk.unsqueeze(1) == sampled_ids,
                torch.full_like(sampled_negatives_logits, -5e4),
                sampled_negatives_logits,
            )
            sampled_logits = torch.cat([positive_logits, sampled_negatives_logits], dim=1)
            total_loss = total_loss + (-F.log_softmax(sampled_logits, dim=1)[:, 0]).sum()

        return total_loss / num_examples

    def _sample_local_negatives(
        self,
        positive_ids: Tensor,
        num_to_sample: int,
    ) -> tuple[Tensor, Tensor]:
        candidate_ids = self._catalog_item_ids
        sampled_ids = torch.randint(
            low=0,
            high=candidate_ids.numel(),
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = candidate_ids[sampled_ids]
        return sampled_ids, self._embedding(sampled_ids)

    def set_item_catalog(self, item_ids: list[int] | Tensor) -> None:
        if not isinstance(item_ids, torch.Tensor):
            item_ids = torch.tensor(item_ids, dtype=torch.long)
        self._catalog_item_ids = item_ids.to(self._embedding.weight.device)
        super().set_item_catalog(self._catalog_item_ids)

    def _encode_history(
        self,
        batch: UserHistoryBatch,
        append_output_slot: bool = False,
    ) -> Tensor:
        batch_size, seq_len = batch.history_ids.shape
        device = batch.history_ids.device
        mask = batch.mask.bool()
        item_emb = self._embedding(batch.history_ids)
        if self._rating_embedding_dim > 0:
            rating_emb = self._rating_embedding(batch.history_ratings)
            item_emb = torch.cat([item_emb, rating_emb], dim=-1)
        timestamps = batch.history_timestamps

        if append_output_slot:
            item_emb = F.pad(item_emb, pad=(0, 0, 0, 1))
            mask = F.pad(mask, pad=(0, 1), value=True)
            timestamps = torch.cat([timestamps, batch.timestamp.unsqueeze(1)], dim=1)
            seq_len = seq_len + 1

        positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
        pos_emb = self._pos_embedding(positions)
        x = item_emb * (item_emb.size(-1) ** 0.5) + pos_emb
        if self._tabular_dim > 0:
            history_features = batch.history_features
            if append_output_slot:
                history_features = F.pad(history_features, pad=(0, 0, 0, 1))
            x = torch.cat([x, history_features], dim=-1)
        x = self._input_dropout(x)

        hidden = x * mask.unsqueeze(-1)
        for block in self._hstu:
            hidden = block(hidden, mask, timestamps)

        hidden = self._final_norm(hidden)
        hidden = self._out_proj(hidden)
        if self._output_postproc == "l2":
            hidden = hidden / torch.clamp(
                torch.linalg.norm(hidden, ord=2, dim=-1, keepdim=True),
                min=self._mol_eps,
            )
        elif self._output_postproc == "ln":
            hidden = F.layer_norm(hidden, normalized_shape=(hidden.size(-1),), eps=self._mol_eps)
        elif self._output_postproc != "none":
            raise ValueError(f"Unknown output_postproc: {self._output_postproc}")
        return hidden

    def forward(self, batch: UserHistoryBatch) -> dict[str, Tensor]:
        hidden = self._encode_history(batch=batch, append_output_slot=False)

        out: dict[str, Tensor] = {}
        logits: Tensor | None = None
        needs_logits = (not self._use_mol) or (
            self._dot_loss_weight > 0.0 and (not self.training or not self._dot_sampled_softmax)
        )
        if needs_logits:
            logits = self._compute_logits(hidden)
            out["logits"] = logits
        if self._use_mol:
            out["retrieval_queries"] = hidden
            if not self.training:
                eval_hidden = self._encode_history(
                    batch=batch,
                    append_output_slot=True,
                )
                last_history_idx = batch.mask.sum(dim=1).long() - 1
                out["next_retrieval_queries"] = eval_hidden[
                    torch.arange(eval_hidden.size(0), device=eval_hidden.device),
                    last_history_idx,
                ]

        if batch.target is not None:
            out["targets"] = batch.target.flatten()
            total_loss = hidden.new_zeros(())

            if self._dot_loss_weight > 0.0:
                if self._dot_sampled_softmax and not self._use_mol:
                    dot_loss = self._compute_sampled_dot_training_loss(hidden, batch)
                    if dot_loss is None:
                        dot_loss = hidden.new_zeros(())
                else:
                    if logits is None:
                        logits = self._compute_logits(hidden)
                        out["logits"] = logits
                    dot_loss = self._compute_dot_loss(logits, batch)
                out["dot_loss"] = dot_loss
                total_loss = total_loss + self._dot_loss_weight * dot_loss

            mol_loss = None
            mol_train_aux_losses: dict[str, Tensor] = {}
            if self._use_mol:
                mol_loss, mol_train_aux_losses = self.compute_mol_training_loss(hidden, batch)
                out.update(mol_train_aux_losses)
            if mol_loss is not None:
                out["mol_loss"] = mol_loss
                total_loss = total_loss + self._mol_loss_weight * mol_loss

            aux_mi_loss = mol_train_aux_losses.get("mi_loss")
            if self._mol_mi_loss_weight > 0.0 and aux_mi_loss is not None:
                total_loss = total_loss + self._mol_mi_loss_weight * aux_mi_loss
            uid_embedding_l2_norm = mol_train_aux_losses.get("uid_embedding_l2_norm")
            if self._mol_uid_embedding_l2_weight > 0.0 and uid_embedding_l2_norm is not None:
                total_loss = total_loss + self._mol_uid_embedding_l2_weight * uid_embedding_l2_norm

            out["loss"] = total_loss
        return out
