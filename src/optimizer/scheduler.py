from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from typing import List
from .builder import SCHEDULERS
from torch.optim import Optimizer
from bisect import bisect_right


@SCHEDULERS.register_module()
class HgCAnScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_lr: float):
        self.max_lr = max_lr
        super(HgCAnScheduler, self).__init__(optimizer)

    def get_lr(self) -> List:
        return [max(self.max_lr, base_lr / (self.last_epoch + 1)) for base_lr in self.base_lrs]


@SCHEDULERS.register_module()
class FCTRScheduler(LRScheduler):
    SCHEDULES = ["step", "multistep", "linear_with_warmup", "all_linear_with_warmup"]

    def __init__(self, optimizer, step_size, schedule, epochs=-1, warm_steps_ratio=0.01, lr_drop=35):
        assert schedule in self.SCHEDULES

        self.step_size = step_size

        self.schedule = schedule
        self.warm_steps_ratio = warm_steps_ratio
        self.lr_drop = lr_drop
        self.epochs = epochs
        self.warmup_steps = round(self.warm_steps_ratio * step_size)

        super(FCTRScheduler, self).__init__(optimizer=optimizer)

    def get_lr(self):
        schedule_fcn = getattr(self, self.schedule + "_schedule")
        gammas = schedule_fcn()
        return [lr * gamma for lr, gamma in zip(self.base_lrs, gammas)]

        # lr_cur = []
        # for opg, lr, gg in zip(self.optimizer, self.base_lrs, gammas):
        #     opg["lr"] = lr * gg
        # return lr_cur

    def step_schedule(self):
        gamma = 0.1 ** (self.epochs // self.lr_drop)
        # text_encoder_gamma = gamma
        return [gamma for _ in range(len(self.base_lrs))]

    def multistep_schedule(self):
        milestones = list(range(self.lr_drop, self.epochs, 50))
        epoch = self._step_count // self.step_size
        gamma = 0.5 ** bisect_right(milestones, epoch)
        # text_encoder_gamma = gamma
        return [gamma for _ in range(len(self.base_lrs))]

    def linear_with_warmup_schedule(self):
        epoch = self._step_count // self.step_size
        gamma = 0.1 ** (epoch // self.lr_drop)
        if self._step_count < self.warmup_steps:
            text_encoder_gamma = float(self._step_count) / float(max(1, self.warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(self.step_size - self._step_count) / float(max(1, self.step_size - self.warmup_steps)),
            )

        return [gamma if idx != 1 else text_encoder_gamma for idx, _ in enumerate(range(len(self.base_lrs)))]

    def all_linear_with_warmup_schedule(self):
        if self._step_count < self.warmup_steps:
            text_encoder_gamma = float(self._step_count) / float(max(1, self.warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(self.step_size - self._step_count) / float(max(1, self.step_size - self.warmup_steps)),
            )
        gamma = text_encoder_gamma
        return [gamma for _ in range(len(self.base_lrs))]
