from torch.optim import lr_scheduler as _lr_scheduler

from ...misc.typing import LRScheduler, Optimizer

__all__ = [
    "load_scheduler",
]


def load_scheduler(
    optimizer: Optimizer,
    **kwargs,
) -> LRScheduler:
    if "epochs" not in kwargs:
        raise ValueError("'epochs' must be specified.")
    warmup_scheduler_name = kwargs.get("warmup_lr_scheduler")
    warmup_scheduler_cfg = kwargs.get("warmup_lr_scheduler_cfg", {})
    main_scheduler_name = kwargs.get("lr_scheduler")
    main_scheduler_cfg = kwargs.get("lr_scheduler_cfg", {})
    # Configure warmup scheduler if specified
    if warmup_scheduler_name:
        warmup_epochs = warmup_scheduler_cfg["total_iters"]
        warmup_scheduler = getattr(_lr_scheduler, warmup_scheduler_name)(
            optimizer=optimizer, **warmup_scheduler_cfg
        )
    else:
        warmup_epochs = 0
        warmup_scheduler = None
    # Configure main scheduler
    if main_scheduler_name:
        if main_scheduler_name == "CosineAnnealingLR":
            main_scheduler_cfg["T_max"] = kwargs["epochs"] - warmup_epochs
        main_scheduler = getattr(_lr_scheduler, main_scheduler_name)(
            optimizer=optimizer, **main_scheduler_cfg
        )
    else:
        # Default to constant learning rate if no scheduler specified
        main_scheduler = _lr_scheduler.StepLR(
            optimizer=optimizer, step_size=kwargs["epochs"], gamma=1.0
        )
    # Combine schedulers if using warmup
    if warmup_scheduler:
        return _lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_scheduler_cfg["total_iters"]],
        )
    return main_scheduler
