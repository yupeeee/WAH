from torch.optim import lr_scheduler as _lr_scheduler

from ...typing import LRScheduler, Optimizer

__all__ = [
    "load_scheduler",
]


def load_scheduler(
    optimizer: Optimizer,
    **kwargs,
) -> LRScheduler:
    assert "epochs" in kwargs.keys()

    if "warmup_lr_scheduler" in kwargs.keys():
        warmup_epochs = kwargs.get("warmup_lr_scheduler_cfg")["total_iters"]
        warmup_lr_scheduler = getattr(_lr_scheduler, kwargs.get("warmup_lr_scheduler"))(
            optimizer=optimizer,
            **kwargs.get("warmup_lr_scheduler_cfg"),
        )
    else:
        warmup_epochs = 0
        warmup_lr_scheduler = None

    if "lr_scheduler" in kwargs.keys():
        if kwargs.get("lr_scheduler") == "CosineAnnealingLR":
            kwargs["lr_scheduler_cfg"]["T_max"] = kwargs.get("epochs") - warmup_epochs

        main_lr_scheduler: LRScheduler = getattr(
            _lr_scheduler, kwargs.get("lr_scheduler")
        )(
            optimizer=optimizer,
            **kwargs.get("lr_scheduler_cfg"),
        )
    else:
        main_lr_scheduler: LRScheduler = _lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=kwargs.get("epochs"),
            gamma=1.0,
        )

    if warmup_lr_scheduler is not None:
        lr_scheduler = _lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[kwargs.get("warmup_lr_scheduler_cfg")["total_iters"]],
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler
