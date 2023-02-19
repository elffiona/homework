from typing import List
import math
from torch.optim.lr_scheduler import _LRScheduler
import warnings

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler that multiplies the learning rate by a factor of 0.5
    whenever the validation loss does not improve for `patience` consecutive epochs.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
    """

    def __init__(self, optimizer, T_start, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_start = T_start
        self.T_mult = T_mult
        self.T_max = T_start
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    #
    # def __init__(self, optimizer, gamma=0, last_epoch=-1):
    #     self.gamma = gamma
    #     super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
    #
    def get_lr(self) -> List[float]:
        """
        exponential
        """

        """
        cosine annealing LR with warm restart
        """
        if self.T_cur == 0:
            return self.base_lrs
        elif self.T_cur == self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.T_cur / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """
        Modified based on the step function of pytorch class _LRScheduler
        """
        # Check current number of epoch
        if epoch is None and self.last_epoch < 0:
            # first epoch, initialize
            epoch = 0

        if epoch is None:
            # Not first epoch, just the start of current epoch
            epoch = self.last_epoch + 1
            # Increment T_cur
            self.T_cur = self.T_cur + 1
            # Check if current epoch number is max
            if self.T_cur >= self.T_max:
                self.T_cur = self.T_cur - self.T_max
                self.T_max = self.T_max * self.T_mult
        else:
            # Running current epoch
            if epoch >= self.T_start:
                # Not the fist round of epochs
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_start
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_start * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_start * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_max = self.T_start * self.T_mult ** (n)
            else:
                self.T_max = self.T_start
                self.T_cur = epoch
            self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
