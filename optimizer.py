# Define Optimizer(SGD/Adam/Adamw) & Schedule
# Setup optimizer
import torch
from omegaconf import OmegaConf


def get_optimizer(model: torch.nn.Module, cfg: OmegaConf):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    return optimizer