from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SIMILARITY_HEAD_DEFAULTS: dict[str, dict[str, Any]] = {
    "mol": {
        "dot_product_dimension": 64,
        "query_groups": 8,
        "item_groups": 4,
        "temperature": 0.05,
        "query_hidden_dim": 512,
        "item_hidden_dim": 0,
        "query_dropout_rate": 0.0,
        "item_dropout_rate": 0.1,
        "query_nonlinearity": "swiglu",
        "item_nonlinearity": "swiglu",
        "uid_embedding_hash_sizes": [6040],
        "uid_dropout_rate": 0.5,
        "softmax_dropout_rate": 0.2,
        "num_train_negatives": 128,
        "query_chunk_size": 64,
        "softmax_temperature": 1.0,
        "retrieval_chunk_size": 512,
        "eps": 1.0e-6,
    },
    "cosine": {
        "projection_dim": 64,
        "query_hidden_dim": 256,
        "item_hidden_dim": 0,
        "dropout_rate": 0.1,
        "num_train_negatives": 128,
        "query_chunk_size": 128,
        "softmax_temperature": 0.05,
        "retrieval_chunk_size": 1024,
        "eps": 1.0e-6,
    },
    "bilinear": {
        "projection_dim": 64,
        "query_hidden_dim": 256,
        "item_hidden_dim": 0,
        "dropout_rate": 0.1,
        "num_train_negatives": 128,
        "query_chunk_size": 128,
        "softmax_temperature": 0.05,
        "retrieval_chunk_size": 1024,
        "eps": 1.0e-6,
    },
    "mlp": {
        "projection_dim": 64,
        "query_hidden_dim": 256,
        "item_hidden_dim": 0,
        "mlp_hidden_dim": 256,
        "dropout_rate": 0.1,
        "num_train_negatives": 128,
        "query_chunk_size": 128,
        "softmax_temperature": 0.05,
        "retrieval_chunk_size": 512,
        "eps": 1.0e-6,
    },
}

SIMILARITY_RUNTIME_DEFAULTS: dict[str, Any] = {
    "loss_weight": 1.0,
    "mi_loss_weight": 0.0,
    "uid_embedding_l2_weight": 0.0,
}


def resolve_similarity_head_config(
    similarity_head: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if similarity_head is None:
        name = "mol"
        params: dict[str, Any] = {}
    else:
        raw = dict(similarity_head)
        name = str(raw.get("name", "mol")).lower()
        params = dict(raw.get("params", {}))
    defaults = SIMILARITY_HEAD_DEFAULTS[name]
    merged = dict(defaults)
    merged.update(params)
    return name, merged


def resolve_similarity_runtime_config(
    similarity_runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(SIMILARITY_RUNTIME_DEFAULTS)
    if similarity_runtime is not None:
        merged.update(dict(similarity_runtime))
    return merged


def _build_activation(nonlinearity: str) -> nn.Module:
    if nonlinearity == "relu":
        return nn.ReLU()
    if nonlinearity == "gelu":
        return nn.GELU()
    if nonlinearity == "swiglu":
        return nn.SiLU()
    raise ValueError(f"Unknown nonlinearity: {nonlinearity}")


def _build_projection(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout_rate: float,
    nonlinearity: str,
) -> nn.Module:
    if hidden_dim > 0:
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            _build_activation(nonlinearity),
            nn.Linear(hidden_dim, output_dim),
        )
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(input_dim, output_dim),
    )


class SimilarityHeadBase(nn.Module):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int,
        item_count: int,
        num_train_negatives: int,
        query_chunk_size: int,
        softmax_temperature: float,
        retrieval_chunk_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self._item_embedding = item_embedding
        self._embedding_dim = embedding_dim
        self._num_train_negatives = num_train_negatives
        self._query_chunk_size = query_chunk_size
        self._softmax_temperature = softmax_temperature
        self._retrieval_chunk_size = retrieval_chunk_size
        self._eps = eps
        self._retrieval_cache: dict[str, tuple[Tensor, Any]] = {}
        self.register_buffer(
            "_catalog_item_ids",
            torch.arange(1, item_count, dtype=torch.long),
            persistent=False,
        )

    @property
    def retrieval_item_count(self) -> int:
        return int(self._catalog_item_ids.numel())

    def set_item_catalog(self, item_ids: list[int] | Tensor) -> None:
        if not isinstance(item_ids, torch.Tensor):
            item_ids = torch.tensor(item_ids, dtype=torch.long)
        self._catalog_item_ids = item_ids.to(self._item_embedding.weight.device)
        self.clear_retrieval_index()

    def clear_retrieval_index(self) -> None:
        self._retrieval_cache.clear()

    def prepare_retrieval_index(self, **kwargs: Any) -> None:
        device = str(self._item_embedding.weight.device)
        if device in self._retrieval_cache:
            return
        item_ids = self._catalog_item_ids.to(self._item_embedding.weight.device)
        item_embeddings = self._item_embedding(item_ids).detach()
        self._retrieval_cache[device] = (item_ids, self._prepare_item_cache(item_embeddings, item_ids))

    def _prepare_item_cache(self, item_embeddings: Tensor, item_ids: Tensor) -> Any:
        return item_embeddings

    def _get_item_cache(self, device: torch.device) -> tuple[Tensor, Any]:
        key = str(device)
        cached = self._retrieval_cache.get(key)
        if cached is None:
            item_ids = self._catalog_item_ids.to(device)
            item_embeddings = self._item_embedding(item_ids).detach()
            cached = (item_ids, self._prepare_item_cache(item_embeddings, item_ids))
            if not self.training:
                self._retrieval_cache[key] = cached
        return cached

    def _sample_local_negatives(self, positive_ids: Tensor, num_to_sample: int) -> tuple[Tensor, Tensor]:
        sampled_offsets = torch.randint(
            low=0,
            high=self._catalog_item_ids.numel(),
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._catalog_item_ids[sampled_offsets]
        return sampled_ids, self._item_embedding(sampled_ids)

    def score_candidates(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_embeddings: Tensor,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError()

    def compute_training_loss(
        self,
        query_embeddings: Tensor,
        target_items: Tensor,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        if target_items.numel() == 0:
            return None, {}

        num_negatives = min(self._num_train_negatives, self._catalog_item_ids.numel())
        total_loss = query_embeddings.new_zeros(())
        aux_sums: dict[str, Tensor] = {}

        for start in range(0, target_items.numel(), self._query_chunk_size):
            end = min(start + self._query_chunk_size, target_items.numel())
            query_chunk = query_embeddings[start:end]
            target_chunk = target_items[start:end]
            user_ids_chunk = user_ids[start:end] if user_ids is not None else None
            positive_embeddings = self._item_embedding(target_chunk).unsqueeze(1)
            positive_logits, aux_losses = self.score_candidates(
                query_embeddings=query_chunk,
                item_ids=target_chunk.unsqueeze(1),
                item_embeddings=positive_embeddings,
                user_ids=user_ids_chunk,
            )
            positive_logits = positive_logits / self._softmax_temperature
            sampled_ids, sampled_negative_embeddings = self._sample_local_negatives(
                positive_ids=target_chunk,
                num_to_sample=num_negatives,
            )
            sampled_negative_logits, _ = self.score_candidates(
                query_embeddings=query_chunk,
                item_ids=sampled_ids,
                item_embeddings=sampled_negative_embeddings,
                user_ids=user_ids_chunk,
            )
            sampled_negative_logits = torch.where(
                target_chunk.unsqueeze(1) == sampled_ids,
                torch.full_like(sampled_negative_logits, -5e4),
                sampled_negative_logits / self._softmax_temperature,
            )
            sampled_logits = torch.cat([positive_logits, sampled_negative_logits], dim=1)
            total_loss = total_loss + (-F.log_softmax(sampled_logits, dim=1)[:, 0]).sum()
            for key, value in aux_losses.items():
                aux_sums[key] = aux_sums.get(key, query_embeddings.new_zeros(())) + value * (end - start)

        aux_means = {key: value / target_items.numel() for key, value in aux_sums.items()}
        return total_loss / target_items.numel(), aux_means

    def compute_batch_training_loss(self, query_embeddings: Tensor, batch) -> tuple[Tensor | None, dict[str, Tensor]]:
        valid = batch.loss_mask.bool()
        positions = valid.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return None, {}
        b_idx = positions[:, 0]
        t_idx = positions[:, 1]
        return self.compute_training_loss(
            query_embeddings=query_embeddings[b_idx, t_idx],
            target_items=batch.target[b_idx, t_idx].long(),
            user_ids=batch.user_id[b_idx].long(),
        )

    def retrieve_topk(
        self,
        query_embeddings: Tensor,
        k: int,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        item_ids, item_cache = self._get_item_cache(query_embeddings.device)
        top_scores = None
        top_ids = None
        for start in range(0, item_ids.numel(), self._retrieval_chunk_size):
            end = min(start + self._retrieval_chunk_size, item_ids.numel())
            chunk_ids = item_ids[start:end]
            chunk_cache = self._slice_item_cache(item_cache, start, end)
            chunk_scores, _ = self._score_from_cache(
                query_embeddings=query_embeddings,
                item_ids=chunk_ids,
                item_cache=chunk_cache,
                user_ids=user_ids,
            )
            if top_scores is None:
                top_scores = chunk_scores
                top_ids = chunk_ids.unsqueeze(0).expand(query_embeddings.size(0), -1)
            else:
                candidate_scores = torch.cat([top_scores, chunk_scores], dim=1)
                candidate_ids = torch.cat([
                    top_ids,
                    chunk_ids.unsqueeze(0).expand(query_embeddings.size(0), -1),
                ], dim=1)
                keep = min(k, candidate_scores.size(1))
                top_idx = candidate_scores.topk(keep, dim=1).indices
                top_scores = candidate_scores.gather(1, top_idx)
                top_ids = candidate_ids.gather(1, top_idx)
        assert top_scores is not None and top_ids is not None
        keep = min(k, top_scores.size(1))
        top_idx = top_scores.topk(keep, dim=1).indices
        return top_scores.gather(1, top_idx), top_ids.gather(1, top_idx)

    def _slice_item_cache(self, item_cache: Any, start: int, end: int) -> Any:
        if isinstance(item_cache, Tensor):
            return item_cache[start:end]
        return tuple(x[start:end] if isinstance(x, Tensor) else x for x in item_cache)

    def _score_from_cache(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_cache: Any,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if isinstance(item_cache, Tensor):
            item_embeddings = item_cache.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
            item_id_batch = item_ids.unsqueeze(0).expand(query_embeddings.size(0), -1)
            return self.score_candidates(query_embeddings, item_id_batch, item_embeddings, user_ids=user_ids)
        raise NotImplementedError()


class MoLHead(SimilarityHeadBase):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int,
        item_count: int,
        dot_product_dimension: int,
        query_groups: int,
        item_groups: int,
        temperature: float,
        query_hidden_dim: int,
        item_hidden_dim: int,
        query_dropout_rate: float,
        item_dropout_rate: float,
        query_nonlinearity: str,
        item_nonlinearity: str,
        uid_embedding_hash_sizes: list[int] | None,
        uid_dropout_rate: float,
        softmax_dropout_rate: float,
        num_train_negatives: int,
        query_chunk_size: int,
        softmax_temperature: float,
        retrieval_chunk_size: int,
        eps: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            item_embedding=item_embedding,
            embedding_dim=embedding_dim,
            item_count=item_count,
            num_train_negatives=num_train_negatives,
            query_chunk_size=query_chunk_size,
            softmax_temperature=softmax_temperature,
            retrieval_chunk_size=retrieval_chunk_size,
            eps=eps,
        )
        self._query_groups = query_groups
        self._item_groups = item_groups
        self._dot_product_dimension = dot_product_dimension
        self._temperature = temperature
        self._uid_hash_size = uid_embedding_hash_sizes[0] if uid_embedding_hash_sizes else None
        self._uid_dim = min(embedding_dim, 32) if self._uid_hash_size else 0
        self._uid_dropout = nn.Dropout(uid_dropout_rate)
        self._uid_embedding = (
            nn.Embedding(self._uid_hash_size, self._uid_dim) if self._uid_hash_size else None
        )
        query_input_dim = embedding_dim + self._uid_dim
        self._query_proj = _build_projection(
            input_dim=query_input_dim,
            output_dim=query_groups * dot_product_dimension,
            hidden_dim=query_hidden_dim,
            dropout_rate=query_dropout_rate,
            nonlinearity=query_nonlinearity,
        )
        self._item_proj = _build_projection(
            input_dim=embedding_dim,
            output_dim=item_groups * dot_product_dimension,
            hidden_dim=item_hidden_dim,
            dropout_rate=item_dropout_rate,
            nonlinearity=item_nonlinearity,
        )
        self._query_gate = nn.Sequential(
            nn.Dropout(softmax_dropout_rate),
            nn.Linear(query_input_dim, query_groups),
        )
        self._item_gate = nn.Sequential(
            nn.Dropout(softmax_dropout_rate),
            nn.Linear(embedding_dim, item_groups),
        )

    def _augment_query(self, query_embeddings: Tensor, user_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        if self._uid_embedding is None or user_ids is None:
            return query_embeddings, query_embeddings
        uid = self._uid_embedding(user_ids % self._uid_hash_size)
        uid = self._uid_dropout(uid)
        augmented = torch.cat([query_embeddings, uid], dim=-1)
        uid_l2 = uid.pow(2).sum(dim=-1).mean()
        return augmented, uid_l2

    def _prepare_item_cache(self, item_embeddings: Tensor, item_ids: Tensor) -> Any:
        item_components = self._item_proj(item_embeddings).view(
            item_embeddings.size(0), self._item_groups, self._dot_product_dimension
        )
        item_gate = self._item_gate(item_embeddings)
        return item_embeddings, item_components, item_gate

    def _score_from_cache(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_cache: Any,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        item_embeddings, item_components, item_gate = item_cache
        item_embeddings = item_embeddings.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
        item_components = item_components.unsqueeze(0).expand(query_embeddings.size(0), -1, -1, -1)
        item_gate = item_gate.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
        item_id_batch = item_ids.unsqueeze(0).expand(query_embeddings.size(0), -1)
        return self._score_components(
            query_embeddings=query_embeddings,
            item_ids=item_id_batch,
            item_embeddings=item_embeddings,
            item_components=item_components,
            item_gate_logits=item_gate,
            user_ids=user_ids,
        )

    def score_candidates(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_embeddings: Tensor,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        item_components = self._item_proj(item_embeddings).view(
            item_embeddings.size(0), item_embeddings.size(1), self._item_groups, self._dot_product_dimension
        )
        item_gate = self._item_gate(item_embeddings)
        return self._score_components(
            query_embeddings=query_embeddings,
            item_ids=item_ids,
            item_embeddings=item_embeddings,
            item_components=item_components,
            item_gate_logits=item_gate,
            user_ids=user_ids,
        )

    def _score_components(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_embeddings: Tensor,
        item_components: Tensor,
        item_gate_logits: Tensor,
        user_ids: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        augmented_query, uid_l2 = self._augment_query(query_embeddings, user_ids)
        query_components = self._query_proj(augmented_query).view(
            query_embeddings.size(0), self._query_groups, self._dot_product_dimension
        )
        query_gate = self._query_gate(augmented_query)
        pair_logits = torch.einsum("bqd,bnid->bqni", query_components, item_components) / self._temperature
        gate_logits = query_gate.unsqueeze(-1).unsqueeze(2) + item_gate_logits.unsqueeze(1)
        gate_weights = gate_logits.flatten(1, 2).softmax(dim=1).view_as(gate_logits)
        scores = (pair_logits * gate_weights).sum(dim=(1, 3))
        aux: dict[str, Tensor] = {}
        if self._uid_embedding is not None and user_ids is not None:
            aux["uid_embedding_l2_norm"] = uid_l2
        entropy = -(gate_weights.clamp_min(self._eps).log() * gate_weights).sum(dim=(1, 2, 3)).mean()
        aux["mi_loss"] = -entropy
        return scores, aux


class CosineSimilarityHead(SimilarityHeadBase):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int,
        item_count: int,
        projection_dim: int,
        query_hidden_dim: int,
        item_hidden_dim: int,
        dropout_rate: float,
        num_train_negatives: int,
        query_chunk_size: int,
        softmax_temperature: float,
        retrieval_chunk_size: int,
        eps: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(item_embedding, embedding_dim, item_count, num_train_negatives, query_chunk_size, softmax_temperature, retrieval_chunk_size, eps)
        self._query_proj = _build_projection(embedding_dim, projection_dim, query_hidden_dim, dropout_rate, "gelu")
        self._item_proj = _build_projection(embedding_dim, projection_dim, item_hidden_dim, dropout_rate, "gelu")

    def _prepare_item_cache(self, item_embeddings: Tensor, item_ids: Tensor) -> Any:
        item_proj = F.normalize(self._item_proj(item_embeddings), dim=-1, eps=self._eps)
        return item_embeddings, item_proj

    def _score_from_cache(self, query_embeddings: Tensor, item_ids: Tensor, item_cache: Any, user_ids: Tensor | None = None):
        _, item_proj = item_cache
        query_proj = F.normalize(self._query_proj(query_embeddings), dim=-1, eps=self._eps)
        scores = torch.matmul(query_proj, item_proj.T)
        return scores, {}

    def score_candidates(self, query_embeddings: Tensor, item_ids: Tensor, item_embeddings: Tensor, user_ids: Tensor | None = None):
        query_proj = F.normalize(self._query_proj(query_embeddings), dim=-1, eps=self._eps)
        item_proj = F.normalize(self._item_proj(item_embeddings), dim=-1, eps=self._eps)
        scores = torch.einsum("bd,bnd->bn", query_proj, item_proj)
        return scores, {}


class BilinearSimilarityHead(SimilarityHeadBase):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int,
        item_count: int,
        projection_dim: int,
        query_hidden_dim: int,
        item_hidden_dim: int,
        dropout_rate: float,
        num_train_negatives: int,
        query_chunk_size: int,
        softmax_temperature: float,
        retrieval_chunk_size: int,
        eps: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(item_embedding, embedding_dim, item_count, num_train_negatives, query_chunk_size, softmax_temperature, retrieval_chunk_size, eps)
        self._query_proj = _build_projection(embedding_dim, projection_dim, query_hidden_dim, dropout_rate, "gelu")
        self._item_proj = _build_projection(embedding_dim, projection_dim, item_hidden_dim, dropout_rate, "gelu")
        self._bilinear = nn.Parameter(torch.empty(projection_dim, projection_dim))
        nn.init.xavier_uniform_(self._bilinear)

    def _prepare_item_cache(self, item_embeddings: Tensor, item_ids: Tensor) -> Any:
        return item_embeddings, self._item_proj(item_embeddings)

    def _score_from_cache(self, query_embeddings: Tensor, item_ids: Tensor, item_cache: Any, user_ids: Tensor | None = None):
        _, item_proj = item_cache
        query_proj = self._query_proj(query_embeddings) @ self._bilinear
        scores = torch.matmul(query_proj, item_proj.T)
        return scores, {}

    def score_candidates(self, query_embeddings: Tensor, item_ids: Tensor, item_embeddings: Tensor, user_ids: Tensor | None = None):
        query_proj = self._query_proj(query_embeddings) @ self._bilinear
        item_proj = self._item_proj(item_embeddings)
        scores = torch.einsum("bd,bnd->bn", query_proj, item_proj)
        return scores, {}


class MLPSimilarityHead(SimilarityHeadBase):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int,
        item_count: int,
        projection_dim: int,
        query_hidden_dim: int,
        item_hidden_dim: int,
        mlp_hidden_dim: int,
        dropout_rate: float,
        num_train_negatives: int,
        query_chunk_size: int,
        softmax_temperature: float,
        retrieval_chunk_size: int,
        eps: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(item_embedding, embedding_dim, item_count, num_train_negatives, query_chunk_size, softmax_temperature, retrieval_chunk_size, eps)
        self._query_proj = _build_projection(embedding_dim, projection_dim, query_hidden_dim, dropout_rate, "gelu")
        self._item_proj = _build_projection(embedding_dim, projection_dim, item_hidden_dim, dropout_rate, "gelu")
        self._scorer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(4 * projection_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def _prepare_item_cache(self, item_embeddings: Tensor, item_ids: Tensor) -> Any:
        return item_embeddings, self._item_proj(item_embeddings)

    def _pair_features(self, query_proj: Tensor, item_proj: Tensor) -> Tensor:
        q = query_proj.unsqueeze(1).expand(-1, item_proj.size(1), -1)
        return torch.cat([q, item_proj, q * item_proj, (q - item_proj).abs()], dim=-1)

    def _score_from_cache(self, query_embeddings: Tensor, item_ids: Tensor, item_cache: Any, user_ids: Tensor | None = None):
        _, item_proj = item_cache
        query_proj = self._query_proj(query_embeddings)
        item_proj = item_proj.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
        pair_features = self._pair_features(query_proj, item_proj)
        scores = self._scorer(pair_features).squeeze(-1)
        return scores, {}

    def score_candidates(self, query_embeddings: Tensor, item_ids: Tensor, item_embeddings: Tensor, user_ids: Tensor | None = None):
        query_proj = self._query_proj(query_embeddings)
        item_proj = self._item_proj(item_embeddings)
        pair_features = self._pair_features(query_proj, item_proj)
        scores = self._scorer(pair_features).squeeze(-1)
        return scores, {}


def build_similarity_head(
    similarity_head: Mapping[str, Any] | None,
    *,
    item_embedding: nn.Embedding,
    embedding_dim: int,
    item_count: int,
) -> tuple[str, SimilarityHeadBase]:
    name, cfg = resolve_similarity_head_config(similarity_head)
    head_cls: type[SimilarityHeadBase]
    if name == "mol":
        head_cls = MoLHead
    elif name == "cosine":
        head_cls = CosineSimilarityHead
    elif name == "bilinear":
        head_cls = BilinearSimilarityHead
    elif name == "mlp":
        head_cls = MLPSimilarityHead
    else:
        raise ValueError(f"Unknown similarity head: {name}")
    return name, head_cls(
        item_embedding=item_embedding,
        embedding_dim=embedding_dim,
        item_count=item_count,
        **cfg,
    )
