from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0)
