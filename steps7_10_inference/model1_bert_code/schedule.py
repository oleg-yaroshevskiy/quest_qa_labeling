import torch
from pytorch_transformers import optimization
from poutyne.framework.callbacks import Callback


class _PyTorchLRSchedulerWrapper(Callback):
    def __init__(self, torch_lr_scheduler, *args, **kwargs):
        super().__init__()
        self.torch_lr_scheduler = torch_lr_scheduler
        self.args = args
        self.kwargs = kwargs
        self.scheduler = None
        self.loaded_state = None

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = self.torch_lr_scheduler(optimizer, *self.args, **self.kwargs)

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None

    def on_batch_end(self, batch, logs):
        self.scheduler.step()

    def load_state(self, f):
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(f, map_location="cpu"))
        else:
            self.loaded_state = torch.load(f, map_location="cpu")

    def save_state(self, f):
        torch.save(self.scheduler.state_dict(), f)


class _TotalStepWrapper(_PyTorchLRSchedulerWrapper):
    def on_train_begin(self, logs):
        if "t_total" not in self.kwargs or self.kwargs["t_total"] is None:
            t_total = self.params["steps"] * self.params["epochs"]
            if self.params["accumulation_steps"] is not None:
                t_total = t_total // self.params["accumulation_steps"]
        else:
            t_total = self.kwargs["t_total"]
        if "t_total" in self.kwargs:
            del self.kwargs["t_total"]
        optimizer = self.model.optimizer
        self.scheduler = self.torch_lr_scheduler(
            optimizer, *self.args, t_total=t_total, **self.kwargs
        )

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None


class ConstantLRSchedule(_PyTorchLRSchedulerWrapper):
    """ Constant learning rate schedule.
    """

    def __init__(self, last_epoch=-1):
        super().__init__(optimization.ConstantLRSchedule, last_epoch=last_epoch)


class WarmupConstantSchedule(_PyTorchLRSchedulerWrapper):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, warmup_steps, last_epoch=-1):
        super().__init__(
            optimization.WarmupConstantSchedule, warmup_steps, last_epoch=last_epoch
        )


class WarmupLinearSchedule(_TotalStepWrapper):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, warmup_steps, t_total=None, last_epoch=-1):
        super().__init__(
            optimization.WarmupLinearSchedule,
            warmup_steps,
            t_total=t_total,
            last_epoch=last_epoch,
        )


class WarmupCosineSchedule(_TotalStepWrapper):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, warmup_steps, t_total=None, cycles=0.5, last_epoch=-1):
        super().__init__(
            optimization.WarmupCosineSchedule,
            warmup_steps,
            t_total=t_total,
            cycles=cycles,
            last_epoch=last_epoch,
        )


class WarmupCosineWithHardRestartsSchedule(_TotalStepWrapper):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """

    def __init__(self, warmup_steps, t_total=None, cycles=1.0, last_epoch=-1):
        super().__init__(
            optimization.WarmupCosineWithHardRestartsSchedule,
            warmup_steps,
            t_total=t_total,
            cycles=cycles,
            last_epoch=last_epoch,
        )
