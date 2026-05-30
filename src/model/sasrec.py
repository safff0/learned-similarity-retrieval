from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import Sequential, functional as F

from src.registry import register


@register("model")
class SASRec(nn.Module):
    def __init__(self, n_feats: int, n_class: int, fc_hidden: int = 512) -> None:
        super().__init__()
        # TODO

    def forward(self, data_object: Tensor, labels: Tensor | None = None, **batch: Any) -> dict[str, Tensor]:
        """
        Model forward method.

        Returns logits and (when labels are present) cross-entropy loss.
        This template has no separate criterion module, so the model is
        responsible for populating ``loss`` in its output dict.

        Args:
            data_object (Tensor): input vector.
            labels (Tensor | None): ground-truth labels.
        Returns:
            output (dict): {"logits": ..., "loss": ...} (loss omitted if no labels).
        """
        logits = self.net(data_object)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out

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
