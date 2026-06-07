import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.datasets.base_dataset import UserHistoryBatch
from src.rails.indexing.mol_top_k import MoLBruteForceTopK
from src.rails.similarities.layers import GeGLU, SwiGLU
from src.rails.similarities.mol.item_embeddings_fns import RecoMoLItemEmbeddingsFn
from src.rails.similarities.mol.query_embeddings_fns import RecoMoLQueryEmbeddingsFn
from src.rails.similarities.mol.similarity_fn import (
    MoLSimilarity,
    SoftmaxDropoutCombiner,
)

DEFAULT_MOL_HEAD_PARAMS: dict[str, Any] = {
    "dot_product_dimension": None,
    "query_groups": 2,
    "item_groups": 2,
    "temperature": 1.0,
    "dot_product_l2_norm": True,
    "eps": 1e-6,
    "autocast_bf16": False,
    "num_train_negatives": 256,
    "query_chunk_size": 1024,
    "softmax_temperature": 1.0,
    "query_dropout_rate": 0.0,
    "query_hidden_dim": 0,
    "uid_embedding_hash_sizes": None,
    "uid_dropout_rate": 0.0,
    "uid_embedding_level_dropout": False,
    "item_dropout_rate": 0.0,
    "item_hidden_dim": 0,
    "query_nonlinearity": "geglu",
    "item_nonlinearity": "geglu",
    "gating_query_hidden_dim": 0,
    "gating_item_hidden_dim": 0,
    "gating_qi_hidden_dim": 0,
    "gating_query_fn": True,
    "gating_item_fn": True,
    "gating_item_dropout_rate": 0.0,
    "gating_qi_dropout_rate": 0.0,
    "softmax_dropout_rate": 0.0,
    "gating_combination_type": "glu_silu",
}

DEFAULT_MOL_RUNTIME_PARAMS: dict[str, Any] = {
    "mi_loss_weight": None,
    "uid_embedding_l2_weight": 0.0,
    "loss_weight": 1.0,
}


def resolve_mol_head_params(mol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    params = dict(DEFAULT_MOL_HEAD_PARAMS)
    if mol is not None:
        params.update(dict(mol))
    return params


def resolve_mol_runtime_params(mol_runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    params = dict(DEFAULT_MOL_RUNTIME_PARAMS)
    if mol_runtime is not None:
        params.update(dict(mol_runtime))
    return params


def _init_mlp_xavier_weights_zero_bias(module: nn.Module) -> nn.Module:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                child.bias.data.fill_(0.0)
    return module


def _build_mol_projection(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout_rate: float,
    nonlinearity: str,
) -> nn.Module:
    if hidden_dim > 0:
        activation = (
            GeGLU(in_features=input_dim, out_features=hidden_dim)
            if nonlinearity == "geglu"
            else SwiGLU(in_features=input_dim, out_features=hidden_dim)
        )
        return _init_mlp_xavier_weights_zero_bias(
            nn.Sequential(
                nn.Dropout(p=dropout_rate),
                activation,
                nn.Linear(hidden_dim, output_dim),
            )
        )

    return _init_mlp_xavier_weights_zero_bias(
        nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim, output_dim),
        )
    )


def _build_gating_projection(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout_rate: float,
    use_bias: bool,
) -> nn.Module:
    if hidden_dim > 0:
        return _init_mlp_xavier_weights_zero_bias(
            nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim, bias=use_bias),
            )
        )
    return _init_mlp_xavier_weights_zero_bias(
        nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim, output_dim, bias=use_bias),
        )
    )


class MoLHead(nn.Module):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int,
        item_count: int,
        dot_product_dimension: int | None = None,
        query_groups: int = 2,
        item_groups: int = 2,
        temperature: float = 1.0,
        dot_product_l2_norm: bool = True,
        eps: float = 1e-6,
        autocast_bf16: bool = False,
        num_train_negatives: int = 256,
        query_chunk_size: int = 1024,
        softmax_temperature: float = 1.0,
        query_dropout_rate: float = 0.0,
        query_hidden_dim: int = 0,
        uid_embedding_hash_sizes: list[int] | None = None,
        uid_dropout_rate: float = 0.0,
        uid_embedding_level_dropout: bool = False,
        item_dropout_rate: float = 0.0,
        item_hidden_dim: int = 0,
        query_nonlinearity: str = "geglu",
        item_nonlinearity: str = "geglu",
        gating_query_hidden_dim: int = 0,
        gating_item_hidden_dim: int = 0,
        gating_qi_hidden_dim: int = 0,
        gating_query_fn: bool = True,
        gating_item_fn: bool = True,
        gating_item_dropout_rate: float = 0.0,
        gating_qi_dropout_rate: float = 0.0,
        softmax_dropout_rate: float = 0.0,
        gating_combination_type: str = "glu_silu",
    ) -> None:
        super().__init__()
        self._item_embedding = item_embedding
        self._embedding_dim = embedding_dim
        self._num_train_negatives = num_train_negatives
        self._query_chunk_size = query_chunk_size
        self._softmax_temperature = softmax_temperature
        self._uid_embedding_hash_sizes = uid_embedding_hash_sizes or []
        self._retrieval_cache: dict[str, nn.Module] = {}
        self.register_buffer(
            "_catalog_item_ids",
            torch.arange(1, item_count, dtype=torch.long),
            persistent=False,
        )

        mol_dim = dot_product_dimension or embedding_dim
        query_embeddings_fn = RecoMoLQueryEmbeddingsFn(
            query_embedding_dim=embedding_dim,
            query_dot_product_groups=query_groups,
            dot_product_dimension=mol_dim,
            dot_product_l2_norm=dot_product_l2_norm,
            proj_fn=lambda input_dim, output_dim: _build_mol_projection(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=query_hidden_dim,
                dropout_rate=query_dropout_rate,
                nonlinearity=query_nonlinearity,
            ),
            eps=eps,
            uid_embedding_hash_sizes=self._uid_embedding_hash_sizes,
            uid_dropout_rate=uid_dropout_rate,
            uid_embedding_level_dropout=uid_embedding_level_dropout,
        )
        item_embeddings_fn = RecoMoLItemEmbeddingsFn(
            item_embedding_dim=embedding_dim,
            item_dot_product_groups=item_groups,
            dot_product_dimension=mol_dim,
            dot_product_l2_norm=dot_product_l2_norm,
            proj_fn=lambda input_dim, output_dim: _build_mol_projection(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=item_hidden_dim,
                dropout_rate=item_dropout_rate,
                nonlinearity=item_nonlinearity,
            ),
            eps=eps,
        )
        self._similarity = MoLSimilarity(
            query_embedding_dim=embedding_dim,
            item_embedding_dim=embedding_dim,
            dot_product_dimension=mol_dim,
            query_dot_product_groups=query_groups,
            item_dot_product_groups=item_groups,
            temperature=temperature,
            dot_product_l2_norm=dot_product_l2_norm,
            query_embeddings_fn=query_embeddings_fn,
            item_embeddings_fn=item_embeddings_fn,
            item_proj_fn=None,
            gating_query_only_partial_fn=(
                (
                    lambda input_dim, output_dim: _build_gating_projection(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=gating_query_hidden_dim,
                        dropout_rate=0.0,
                        use_bias=False,
                    )
                )
                if gating_query_fn
                else None
            ),
            gating_item_only_partial_fn=(
                (
                    lambda input_dim, output_dim: _build_gating_projection(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=gating_item_hidden_dim,
                        dropout_rate=gating_item_dropout_rate,
                        use_bias=False,
                    )
                )
                if gating_item_fn
                else None
            ),
            gating_qi_partial_fn=lambda input_dim, output_dim: _build_gating_projection(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=gating_qi_hidden_dim,
                dropout_rate=gating_qi_dropout_rate,
                use_bias=True,
            ),
            gating_combination_type=gating_combination_type,
            gating_normalization_fn=lambda _: SoftmaxDropoutCombiner(
                dropout_rate=softmax_dropout_rate,
                eps=eps,
            ),
            eps=eps,
            apply_query_embeddings_fn=True,
            apply_item_embeddings_fn=True,
            autocast_bf16=autocast_bf16,
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

    def similarity_fn(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_embeddings: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if item_embeddings is None:
            item_embeddings = self._item_embedding(item_ids)
        return self._similarity(
            query_embeddings=query_embeddings,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            **kwargs,
        )

    def _sample_local_negatives(
        self,
        positive_ids: Tensor,
        num_to_sample: int,
    ) -> tuple[Tensor, Tensor]:
        sampled_offsets = torch.randint(
            low=0,
            high=self._catalog_item_ids.numel(),
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._catalog_item_ids[sampled_offsets]
        return sampled_ids, self._item_embedding(sampled_ids)

    def compute_training_loss(
        self,
        query_embeddings: Tensor,
        target_items: Tensor,
        user_ids: Tensor | None = None,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        if target_items.numel() == 0:
            return None, {}

        num_negatives = min(self._num_train_negatives, self._catalog_item_ids.numel())
        positive_embeddings = self._item_embedding(target_items)
        total_loss = query_embeddings.new_zeros(())
        aux_sums: dict[str, Tensor] = {}

        for start in range(0, target_items.numel(), self._query_chunk_size):
            end = min(start + self._query_chunk_size, target_items.numel())
            query_chunk = query_embeddings[start:end]
            target_chunk = target_items[start:end]
            positive_chunk = positive_embeddings[start:end]
            user_ids_chunk = user_ids[start:end] if user_ids is not None else None

            positive_logits, aux_losses = self.similarity_fn(
                query_embeddings=query_chunk,
                item_ids=target_chunk.unsqueeze(1),
                item_embeddings=positive_chunk.unsqueeze(1),
                user_ids=user_ids_chunk,
            )
            positive_logits = positive_logits / self._softmax_temperature
            sampled_ids, sampled_negative_embeddings = self._sample_local_negatives(
                positive_ids=target_chunk,
                num_to_sample=num_negatives,
            )
            sampled_negatives_logits, _ = self.similarity_fn(
                query_embeddings=query_chunk,
                item_ids=sampled_ids,
                item_embeddings=sampled_negative_embeddings,
                user_ids=user_ids_chunk,
            )
            sampled_negatives_logits = torch.where(
                target_chunk.unsqueeze(1) == sampled_ids,
                torch.full_like(sampled_negatives_logits, -5e4),
                sampled_negatives_logits / self._softmax_temperature,
            )
            sampled_logits = torch.cat([positive_logits, sampled_negatives_logits], dim=1)
            total_loss = total_loss + (-F.log_softmax(sampled_logits, dim=1)[:, 0]).sum()
            for key, value in aux_losses.items():
                aux_sums[key] = aux_sums.get(key, query_embeddings.new_zeros(())) + value * (
                    end - start
                )

        aux_means = {key: value / target_items.numel() for key, value in aux_sums.items()}
        return total_loss / target_items.numel(), aux_means

    def compute_batch_training_loss(
        self,
        query_embeddings: Tensor,
        batch: UserHistoryBatch,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        valid = batch.loss_mask.bool()
        positions = valid.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return None, {}

        batch_indices = positions[:, 0]
        time_indices = positions[:, 1]
        return self.compute_training_loss(
            query_embeddings=query_embeddings[batch_indices, time_indices],
            target_items=batch.target[batch_indices, time_indices].long(),
            user_ids=batch.user_id[batch_indices].long(),
        )

    def retrieve_topk(
        self,
        query_embeddings: Tensor,
        k: int,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        cache_key = str(query_embeddings.device)
        topk_module = None
        if not self.training:
            topk_module = self._retrieval_cache.get(cache_key)
        if topk_module is None:
            item_ids = self._catalog_item_ids.to(query_embeddings.device).unsqueeze(0)
            item_embeddings = self._item_embedding(item_ids).detach()
            topk_module = MoLBruteForceTopK(
                mol_module=self._similarity,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            )
            if not self.training:
                self._retrieval_cache[cache_key] = topk_module
        return topk_module(query_embeddings=query_embeddings, k=k, **kwargs)

    def prepare_retrieval_index(
        self,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = str(self._item_embedding.weight.device)
        cache_key = device
        if cache_key in self._retrieval_cache:
            return
        dummy_query = torch.zeros(
            1,
            self._embedding_dim,
            device=self._item_embedding.weight.device,
            dtype=self._item_embedding.weight.dtype,
        )
        self.retrieve_topk(
            query_embeddings=dummy_query,
            k=1,
            user_ids=(
                torch.zeros(1, dtype=torch.long, device=self._item_embedding.weight.device)
                if self._uid_embedding_hash_sizes
                else None
            ),
        )
