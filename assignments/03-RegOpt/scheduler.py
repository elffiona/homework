import math
from typing import List
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts


class CustomLRScheduler(_LRScheduler):
    """
    Implemented Scheduler
    """

    def __init__(self, optimizer, last_epoch=-1, decay_factor=0.5, decay_epochs=20):
        """
        Create a new scheduler that implements step decay.
        decay_factor: factor by which the learning rate will be reduced after each decay_epochs
        decay_epochs: the number of epochs after which the learning rate will be decayed.
        """
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute the new learning rate after each epoch.
        """
        new_lr = []
        for lr in self.base_lrs:
            new_lr.append(
                lr * self.decay_factor ** (self.last_epoch // self.decay_epochs)
            )
        return new_lr
