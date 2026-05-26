import sys

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.datasets import build_dataloaders
from src.metrics import build_metrics
from src.model import build_model
from src.trainer import Trainer, build_optimizer
from src.utils.init_utils import get_device, load_config, set_seed


def main() -> None:
    cfg = load_config(sys.argv[1:])
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    loaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    metrics = build_metrics(cfg)
    writer = SummaryWriter(cfg.trainer.tb_dir)

    print(model)
    Trainer(model, optimizer, loaders, metrics, cfg, writer, device).train()


if __name__ == "__main__":
    main()
