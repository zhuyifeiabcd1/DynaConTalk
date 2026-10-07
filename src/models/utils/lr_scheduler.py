import numpy as np
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Epoch-wise cosine schedule, evaluated in closed form from the current epoch.

    ``t_max_schedule`` is a list of ``[epoch, T_max]`` pairs: from ``epoch`` on, the
    cosine period becomes ``T_max`` (still evaluated at the absolute epoch, so the
    learning rate drops at the switch point). This reproduces the released runs,
    which shortened the period partway through training.
    """

    def __init__(self, optimizer, T_max, warmup=0, eta_min=1e-5, t_max_schedule=None):
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        self.t_max_schedule = sorted((int(e), float(t)) for e, t in (t_max_schedule or []))
        super().__init__(optimizer)

    def current_t_max(self):
        t_max = self.T_max
        for epoch, value in self.t_max_schedule:
            if self.last_epoch >= epoch:
                t_max = value
        return t_max

    def get_lr(self):
        t_max = self.current_t_max()
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch >= self.warmup:
                lrs.append(self.eta_min
                           + (base_lr - self.eta_min) *
                           (1 + np.cos(np.pi * (self.last_epoch - self.warmup) / (t_max - self.warmup))) / 2)
            else:
                lrs.append(base_lr * (self.last_epoch * 1.0 / self.warmup))
        return lrs
