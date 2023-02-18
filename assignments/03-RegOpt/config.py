from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    """
    configs class
    """

    batch_size = 128
    num_epochs = 30
    initial_learning_rate = 0.1
    initial_weight_decay = 0.0005
    momentum = 0

    lrs_kwargs = {"T_start": 10, "T_mult": 2, "eta_min": 0}

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
        momentum=CONFIG.momentum,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )

    train_transforms = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
