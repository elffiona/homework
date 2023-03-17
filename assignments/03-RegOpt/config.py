from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    """
    configs class
    """

    batch_size = 64
    num_epochs = 20
    initial_learning_rate = 0.1
    initial_weight_decay = 0.0005
    momentum = 0.1

    lrs_kwargs = {
        "T_start": 3,
        "T_mult": 1.5,
        "eta_min": 0,
        "num_batch": 782,
    }
    # lrs_kwargs = {
    #     "gamma": 0.95,
    #     "num_batch": 391,
    # }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
        momentum=CONFIG.momentum,
    )
    # optimizer_factory: Callable[
    #     [nn.Module], torch.optim.Optimizer
    # ] = lambda model: torch.optim.Adam(
    #     model.parameters(),
    #     lr=CONFIG.initial_learning_rate,
    #     weight_decay=CONFIG.initial_weight_decay,
    # )

    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
