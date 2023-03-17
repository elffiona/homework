from typing import Callable
import torch
import torch.optim
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    Normalize,
)


class CONFIG:
    """configuration"""

    batch_size = 85
    num_epochs = 15

    optimizer_factory: Callable[
        [torch.nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)

    transforms_train = Compose(
        [
            RandomCrop(size=(32, 32), padding=4),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    transforms_test = Compose(
        [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    transforms = Compose([ToTensor()])
