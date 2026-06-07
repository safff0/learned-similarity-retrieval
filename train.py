import sys
import warnings
import logging

from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
import torch

import src.datasets  # noqa: F401 — triggers @register decorators
import src.metrics  # noqa: F401
import src.model  # noqa: F401
from src.datasets.data_utils import get_dataloaders
from src.metrics import build_metrics
from src.registry import get
from src.trainer import Trainer, build_optimizer
from src.utils.init_utils import load_config, resolve_device, set_random_seed, init_logging

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def _load_init_checkpoint(model, checkpoint_path: str, strict: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if strict:
        model.load_state_dict(state_dict, strict=True)
        logger.info("initialized model from checkpoint %s (strict)", checkpoint_path)
        return

    current_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing = sorted(set(current_state) - set(filtered_state))
    skipped = sorted(set(state_dict) - set(filtered_state))
    model.load_state_dict(filtered_state, strict=False)
    logger.info(
        "initialized model from checkpoint %s with %d matched tensors, %d missing, %d skipped",
        checkpoint_path,
        len(filtered_state),
        len(missing),
        len(skipped),
    )


def main() -> None:
    """
    Main training entrypoint. Loads the YAML config (with CLI dot overrides),
    builds the model / optimizer / metrics / dataloaders, and runs Trainer.
    """
    torch.cuda.empty_cache()
    init_logging()
    config = load_config(sys.argv[1:])

    set_random_seed(config.trainer.seed)
    device = resolve_device(config.trainer.device)

    dataloaders = get_dataloaders(config, device)
    logger.info("loaded datasets")

    config.model.params["item_count"] = dataloaders["train"].dataset.item_count
    model = build("model", config.model).to(device)
    logger.info(f"loaded model: {config.model.name} (item_count={config.model.params['item_count']})")

    train_dataset = getattr(dataloaders["train"], "dataset", None)
    if hasattr(model, "set_item_catalog") and train_dataset is not None and hasattr(train_dataset, "all_item_ids"):
        model.set_item_catalog(train_dataset.all_item_ids)
        logger.info("loaded item catalog with %d ids", len(train_dataset.all_item_ids))

    if init_checkpoint:
        _load_init_checkpoint(model, init_checkpoint, strict=init_strict)

    metrics = build_metrics(config)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(config, trainable_params)

    writer = SummaryWriter(config.trainer.tb_dir)

    epoch_len = config.trainer.epoch_len

    trainer = Trainer(
        model=model,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        writer=writer,
        epoch_len=epoch_len,
    )

    trainer.train()


if __name__ == "__main__":
    main()
