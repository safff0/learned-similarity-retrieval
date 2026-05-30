from abc import abstractmethod

from torch import nn, Tensor

from src.registry import register
from src.datasets.base_dataset import UserHistoryBatch


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, batch: UserHistoryBatch) -> dict[str, Tensor]:
        pass

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
