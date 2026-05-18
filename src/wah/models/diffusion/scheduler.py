from typing import Dict, List, Tuple

import diffusers

__all__ = [
    "SchedulerNames",
    "load_scheduler",
]


def _init_scheduler_map() -> Tuple[List[str], Dict[str, diffusers.SchedulerMixin]]:
    scheduler_names = [
        v.replace("scheduling_", "")
        for v in dir(diffusers.schedulers)
        if "scheduling_" in v and "utils" not in v
    ]

    scheduler_map = dict.fromkeys(scheduler_names)
    for name in scheduler_names:
        f = getattr(diffusers.schedulers, f"scheduling_{name}")
        candidates = [v for v in dir(f) if "Scheduler" in v and "Output" not in v]
        candidates = [
            v
            for v in candidates
            if v not in ["KarrasDiffusionSchedulers", "SchedulerMixin"]
        ]
        assert len(candidates) == 1
        scheduler_cls_name = candidates[0]
        scheduler_map[name] = getattr(f, scheduler_cls_name)

    return scheduler_names, scheduler_map


SchedulerNames, _SchedulerMap = _init_scheduler_map()


def load_scheduler(
    name: str,
    pipe,
) -> None:
    scheduler_cls = _SchedulerMap[name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
