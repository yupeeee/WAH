import diffusers
from diffusers import SchedulerMixin

from ...misc.typing import Dict, Literal

__all__ = [
    "load_scheduler",
]


def load_scheduler(
    version: str,
    model_ids: Dict[str, str],
    strategy: Literal[
        "DDIM",
        "LMSDiscrete",
        "EulerDiscrete",
        "EulerAncestralDiscrete",
        "DPMSolverMultistep",
    ] = "DDIM",
    # **kwargs,
) -> SchedulerMixin:
    scheduler = getattr(diffusers, f"{strategy}Scheduler").from_config(
        model_ids[version],
        subfolder="scheduler",
        # **kwargs,
    )
    return scheduler
