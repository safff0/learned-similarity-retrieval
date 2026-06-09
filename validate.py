import dataclasses
import logging
import sys

from omegaconf import OmegaConf
import torch

import src.datasets  # noqa: F401  triggers @register
import src.metrics   # noqa: F401
import src.model     # noqa: F401
from src.datasets.data_utils import get_dataloaders
from src.metrics.hitrate_full import HitrateFull
from src.metrics.hitrate_mol import HitrateMoL
from src.metrics.mrr_full import MRRFull
from src.metrics.mrr_mol import MRRMoL
from src.metrics.ndcg_full import NDCGFull
from src.metrics.ndcg_mol import NDCGMoL
from src.metrics.tracker import MetricTracker
from src.registry import build
from src.utils.init_utils import (
    init_logging,
    load_config,
    load_init_checkpoint,
    resolve_device,
)

logger = logging.getLogger(__name__)

K_VALUES = (10, 50, 100, 200)


def _move_to_device(batch, device):
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if isinstance(v, torch.Tensor):
            setattr(batch, f.name, v.to(device))
    return batch


def main() -> None:
    init_logging()
    config = load_config(sys.argv[1:])
    device = resolve_device(config.trainer.device)

    dataloaders = get_dataloaders(config, device)
    val_loader = dataloaders["val"]
    val_dataset = val_loader.dataset

    model_params = dict(config.model.params)
    init_checkpoint = model_params.pop("init_checkpoint", None)
    init_strict = bool(model_params.pop("init_strict", True))
    if hasattr(val_dataset, "all_item_ids"):
        model_params.setdefault("item_count", len(val_dataset.all_item_ids) + 1)

    if not init_checkpoint:
        raise ValueError(
            "validate.py requires a checkpoint. Pass "
            "`model.params.init_checkpoint=path/to/checkpoint.pth` on the CLI."
        )

    model = build(
        "model",
        OmegaConf.create({"name": config.model.name, "params": model_params}),
    ).to(device)
    load_init_checkpoint(model, init_checkpoint, strict=init_strict)
    model.eval()
    logger.info("loaded checkpoint: %s", init_checkpoint)

    use_mol_path = hasattr(model, "retrieve_topk")
    metrics = []
    if use_mol_path:
        for k in K_VALUES:
            metrics.append(HitrateMoL(k=k, last_only=True, alias=f"HR@{k}"))
            metrics.append(NDCGMoL(k=k, last_only=True, alias=f"NDCG@{k}"))
        metrics.append(MRRMoL(last_only=True, alias="MRR"))
    else:
        for k in K_VALUES:
            metrics.append(HitrateFull(k=k, last_only=True, alias=f"HR@{k}"))
            metrics.append(NDCGFull(k=k, last_only=True, alias=f"NDCG@{k}"))
        metrics.append(MRRFull(last_only=True, alias="MRR"))
    logger.info("metric family: %s", "MoL (retrieve_topk)" if use_mol_path else "full-vocab")

    tracker = MetricTracker(*[m.alias for m in metrics])

    for met in metrics:
        if hasattr(met, "prepare"):
            met.prepare(model=model)

    with torch.no_grad():
        for batch in val_loader:
            batch = _move_to_device(batch, device)
            outputs = model(batch)
            flat = {**vars(batch), **outputs, "model": model}
            for met in metrics:
                value, n = met(**flat)
                if n > 0:
                    tracker.update(met.alias, value, n=n)

    for met in metrics:
        if hasattr(met, "cleanup"):
            met.cleanup(model=model)

    print(f"\n=== validate.py: {init_checkpoint} ===\n")
    width = max(len(k) for k in tracker.keys())
    for key in tracker.keys():
        print(f"  {key:<{width}}  {tracker.avg(key):.4f}")
    print()


if __name__ == "__main__":
    main()
