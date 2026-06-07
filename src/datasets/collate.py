import torch
from torch.nn.utils.rnn import pad_sequence

from src.datasets.base_dataset import UserHistoryBatch, UserHistoryItem

PAD_ID = 0


def collate_fn(dataset_items: list[UserHistoryItem]) -> UserHistoryBatch:
    history_ids = pad_sequence(
        [it.history_ids for it in dataset_items],
        batch_first=True,
        padding_value=PAD_ID,
    )
    history_ratings = pad_sequence(
        [it.history_ratings for it in dataset_items],
        batch_first=True,
        padding_value=0,
    )
    history_timestamps = pad_sequence(
        [it.history_timestamps for it in dataset_items],
        batch_first=True,
        padding_value=0,
    )
    history_features = pad_sequence(
        [it.history_features for it in dataset_items],
        batch_first=True,
        padding_value=0.0,
    )
    target = pad_sequence(
        [it.target for it in dataset_items],
        batch_first=True,
        padding_value=PAD_ID,
    )
    target_feedback = pad_sequence(
        [it.target_feedback for it in dataset_items],
        batch_first=True,
        padding_value=0,
    )
    loss_mask = pad_sequence(
        [it.loss_mask for it in dataset_items],
        batch_first=True,
        padding_value=False,
    )
    mask = history_ids != PAD_ID

    return UserHistoryBatch(
        user_id=torch.tensor([it.user_id for it in dataset_items], dtype=torch.long),
        history_ids=history_ids,
        history_ratings=history_ratings,
        history_timestamps=history_timestamps,
        history_features=history_features,
        target=target,
        target_feedback=target_feedback,
        mask=mask,
        loss_mask=loss_mask,
        timestamp=torch.tensor([it.timestamp for it in dataset_items], dtype=torch.long),
    )
