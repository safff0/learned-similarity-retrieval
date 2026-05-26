import sys
import warnings

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import src.datasets  # noqa: F401 — triggers @register decorators
import src.metrics  # noqa: F401
import src.model  # noqa: F401
from src.datasets.data_utils import get_dataloaders
from src.metrics import build_metrics
from src.registry import build
from src.trainer import Trainer, build_optimizer
from src.utils.init_utils import load_config, resolve_device, set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """
    Main training entrypoint. Loads the YAML config (with CLI dot overrides),
    builds the model / optimizer / metrics / dataloaders, and runs Trainer.
    """
    config = load_config(sys.argv[1:])
    print(OmegaConf.to_yaml(config))

    set_random_seed(config.trainer.seed)
    device = resolve_device(config.trainer.device)

    dataloaders = get_dataloaders(config, device)

    model = build("model", config.model).to(device)
    print(model)

    metrics = build_metrics(config)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(config, trainable_params)

    writer = SummaryWriter(config.trainer.tb_dir)

    epoch_len = config.trainer.get("epoch_len")

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
