import os
from typing import Literal, Optional

import torch

from . import pipelines
from .pipelines import Pipelines
from .scheduler import SchedulerNames, load_scheduler

__all__ = [
    "load_pipe",
]


def load_pipe(
    name: str,
    scheduler: str = None,
    torch_dtype=torch.float32,
    use_safetensors: bool = True,
    compile_pipe: Optional[Literal["reduce-overhead", "max-autotune"]] = None,
    **kwargs,
):
    assert (
        name in Pipelines
    ), f"Unsupported pipeline: {name} (must be one of {', '.join(list(Pipelines.keys()))})"

    structure, model_name = Pipelines[name]
    utils = getattr(pipelines, structure)

    pipe = getattr(utils, "_Pipeline").from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
        cache_dir=os.environ["HF_HUB_CACHE"],
        token=os.environ["HF_TOKEN"],
        **kwargs,
    )
    pipe.set_progress_bar_config(disable=True)

    if compile_pipe is not None:
        noise_predictor = getattr(utils, "_NoisePredictorName")
        getattr(pipe, noise_predictor).to(memory_format=torch.channels_last)
        setattr(
            pipe,
            noise_predictor,
            torch.compile(
                getattr(pipe, noise_predictor),
                mode=compile_pipe,
                fullgraph=True,
            ),
        )

    if scheduler is not None:
        assert (
            scheduler in SchedulerNames
        ), f"Unsupported scheduler: {scheduler} (must be one of {', '.join(SchedulerNames)})"
        load_scheduler(scheduler, pipe)

    return pipe, utils
