import math
import torch


def setup_optimizer(optim_cfg, model):
    """
    Setup the optimizer. Return the optimizer.
    """
    optimizer = eval(optim_cfg.name)
    model_trainable_params = get_named_trainable_params(model)
    # Print size of trainable parameters
    print(
        "Trainable parameters:",
        sum(p.numel() for (name, p) in model_trainable_params) / 1e6,
        "M",
    )
    return optimizer(list(model.parameters()), **optim_cfg.kwargs)


def setup_lr_scheduler(optimizer, scheduler_cfg):
    sched = eval(scheduler_cfg.name)
    if sched is None:
        return None
    return sched(optimizer, **scheduler_cfg.kwargs)


def get_named_trainable_params(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


class CosineAnnealingLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_lr, warmup_epoch, T_max, last_epoch=-1):
        self.warmup_lr = warmup_lr
        self.warmup_epoch = warmup_epoch
        self.T_max = T_max
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for i in range(len(self.base_lrs)):
            if self.last_epoch < self.warmup_epoch:
                lr = (
                    self.warmup_lr
                    + (self.base_lrs[i] - self.warmup_lr)
                    * self.last_epoch
                    / self.warmup_epoch
                )
            else:
                lr = (
                    0.5
                    * self.base_lrs[i]
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_epoch)
                            / (self.T_max - self.warmup_epoch)
                        )
                    )
                )
            lrs.append(lr)
        return lrs
    

    
