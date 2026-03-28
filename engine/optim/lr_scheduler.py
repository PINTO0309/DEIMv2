"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import math
from functools import partial


def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    """
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.

    Args:
        total_iter (int): Total number of iterations.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_iter (int): Number of iterations for flat phase.
        no_aug_iter (int): Number of iterations for no-augmentation phase.
        current_iter (int): Current iteration.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: Calculated learning rate.
    """
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /
                                           (total_iter - flat_iter - no_aug_iter)))
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler:
    """
    Learning rate scheduler with warm-up, optional flat phase, and cosine decay following RTMDet.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        lr_gamma (float): Scaling factor for the minimum learning rate.
        iter_per_epoch (int): Number of iterations per epoch.
        total_epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of warm-up epochs.
        flat_epochs (int): Number of flat epochs (for flat-cosine scheduler).
        no_aug_epochs (int): Number of no-augmentation epochs.
    """
    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs, 
                 warmup_iter, flat_epochs, no_aug_epochs, scheduler_type="cosine"):
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        self.lr_gamma = lr_gamma
        self.iter_per_epoch = iter_per_epoch
        self.total_epochs = total_epochs
        self.warmup_iter = warmup_iter
        self.flat_epochs = flat_epochs
        self.no_aug_epochs = no_aug_epochs
        self.scheduler_type = scheduler_type

        total_iter = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)

        print(self.base_lrs, self.min_lrs, total_iter, warmup_iter, flat_iter, no_aug_iter)
        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)

    def step(self, current_iter, optimizer):
        """
        Updates the learning rate of the optimizer at the current iteration.

        Args:
            current_iter (int): Current iteration.
            optimizer (torch.optim.Optimizer): Optimizer instance.
        """
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
        return optimizer

    def state_dict(self):
        return {
            'base_lrs': list(self.base_lrs),
            'min_lrs': list(self.min_lrs),
            'lr_gamma': self.lr_gamma,
            'iter_per_epoch': self.iter_per_epoch,
            'total_epochs': self.total_epochs,
            'warmup_iter': self.warmup_iter,
            'flat_epochs': self.flat_epochs,
            'no_aug_epochs': self.no_aug_epochs,
            'scheduler_type': self.scheduler_type,
        }

    def load_state_dict(self, state_dict):
        self.base_lrs = list(state_dict['base_lrs'])
        self.min_lrs = list(state_dict['min_lrs'])
        self.lr_gamma = state_dict['lr_gamma']
        self.iter_per_epoch = state_dict['iter_per_epoch']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_iter = state_dict['warmup_iter']
        self.flat_epochs = state_dict['flat_epochs']
        self.no_aug_epochs = state_dict['no_aug_epochs']
        self.scheduler_type = state_dict.get('scheduler_type', 'cosine')

        total_iter = int(self.iter_per_epoch * self.total_epochs)
        no_aug_iter = int(self.iter_per_epoch * self.no_aug_epochs)
        flat_iter = int(self.iter_per_epoch * self.flat_epochs)
        self.lr_func = partial(flat_cosine_schedule, total_iter, self.warmup_iter, flat_iter, no_aug_iter)
