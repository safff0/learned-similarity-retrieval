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
        return 0

    def retrieve_topk(self, *args, **kwargs):
        mol_head = self._get_mol_head()
        if mol_head is not None:
            return mol_head.retrieve_topk(*args, **kwargs)
        similarity_head = self._get_similarity_head()
        if similarity_head is not None:
            return similarity_head.retrieve_topk(*args, **kwargs)
        raise RuntimeError("retrieve_topk() is only available when a retrieval head is configured")

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
