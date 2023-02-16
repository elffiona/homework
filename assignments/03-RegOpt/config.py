from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor
from scheduler import CustomLRScheduler


class CONFIG:
    batch_size = 80
    num_epochs = 50
    initial_learning_rate = 0.5
    initial_weight_decay = 0.01

    lrs_kwargs = {
        "decay_factor": 0.5,
        "decay_epochs": 10,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        momentum=0.9,
        weight_decay=CONFIG.initial_weight_decay,
    )
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
