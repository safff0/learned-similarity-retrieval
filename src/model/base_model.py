from abc import abstractmethod
from typing import Any

from torch import nn, Tensor

from src.registry import register
from src.datasets.base_dataset import UserHistoryBatch


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, batch: UserHistoryBatch) -> dict[str, Tensor]:
        pass

    def _get_mol_head(self):
        return getattr(self, "_mol_head", None)

    def _get_similarity_head(self):
        return getattr(self, "_similarity_head", None)

    def compute_mol_training_loss(
        self,
        query_embeddings: Tensor,
        batch: UserHistoryBatch,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        mol_head = self._get_mol_head()
        if mol_head is None:
            return None, {}
        return mol_head.compute_batch_training_loss(
            query_embeddings=query_embeddings,
            batch=batch,
        )

    def compute_similarity_training_loss(
        self,
        query_embeddings: Tensor,
        batch: UserHistoryBatch,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        similarity_head = self._get_similarity_head()
        if similarity_head is None:
            return None, {}
        return similarity_head.compute_batch_training_loss(
            query_embeddings=query_embeddings,
            batch=batch,
        )

    def set_item_catalog(self, item_ids) -> None:
        mol_head = self._get_mol_head()
        if mol_head is not None:
            mol_head.set_item_catalog(item_ids)
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            similarity_head.set_item_catalog(item_ids)

    def prepare_retrieval_index(self, **kwargs) -> None:
        mol_head = self._get_mol_head()
        if mol_head is not None:
            mol_head.prepare_retrieval_index()
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            similarity_head.prepare_retrieval_index(**kwargs)

    def clear_retrieval_index(self) -> None:
        mol_head = self._get_mol_head()
        if mol_head is not None:
            mol_head.clear_retrieval_index()
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            similarity_head.clear_retrieval_index()

    @property
    def retrieval_item_count(self) -> int:
        mol_head = self._get_mol_head()
        if mol_head is not None:
            return mol_head.retrieval_item_count
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            return similarity_head.retrieval_item_count
        # Dot-product fallback: number of items in the embedding table (excl. PAD at 0).
        item_emb = getattr(self, "_embedding", None)
        if item_emb is not None:
            return item_emb.num_embeddings - 1
        return 0

    def retrieve_topk(
        self,
        query_embeddings: Tensor,
        k: int,
        user_ids: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        mol_head = self._get_mol_head()
        if mol_head is not None:
            return mol_head.retrieve_topk(
                query_embeddings=query_embeddings,
                k=k,
                user_ids=user_ids,
                **kwargs,
            )
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            return similarity_head.retrieve_topk(
                query_embeddings=query_embeddings,
                k=k,
                user_ids=user_ids,
                **kwargs,
            )
        # Dot-product fallback for vanilla models: score query against every
        # row of the item embedding table and pick top-k. PAD (id 0) is masked.
        item_emb = getattr(self, "_embedding", None)
        if item_emb is None:
            raise RuntimeError(
                "retrieve_topk() needs either a configured head or a `_embedding` "
                "attribute exposing the item embedding table."
            )
        scores = query_embeddings @ item_emb.weight.T  # (B, V)
        scores[:, 0] = float("-inf")
        k = min(k, scores.size(1))
        return scores.topk(k, dim=-1)

    def similarity_fn(
        self,
        query_embeddings: Tensor,
        item_ids: Tensor,
        item_embeddings: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        mol_head = self._get_mol_head()
        if mol_head is not None:
            return mol_head.similarity_fn(
                query_embeddings=query_embeddings,
                item_ids=item_ids,
                item_embeddings=item_embeddings,
                **kwargs,
            )
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            return similarity_head.score_candidates(
                query_embeddings=query_embeddings,
                item_ids=item_ids,
                item_embeddings=item_embeddings,
                **kwargs,
            )
        raise RuntimeError("similarity_fn() is only available when a similarity head is configured")

    def __str__(self) -> str:
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
