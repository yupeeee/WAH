# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
from diffusers import StableDiffusionPipeline

from .utils import is_valid_version, load_scheduler

__all__ = [
    "_load_pipe",
]

model_ids = {
    "1.1": "CompVis/stable-diffusion-v1-1",
    "1.2": "CompVis/stable-diffusion-v1-2",
    "1.3": "CompVis/stable-diffusion-v1-3",
    "1.4": "CompVis/stable-diffusion-v1-4",
    "1.5": "sd-legacy/stable-diffusion-v1-5",
}


def _load_pipe(
    version: str,
    scheduler: str,
    **kwargs,
) -> StableDiffusionPipeline:
    assert is_valid_version(version, model_ids)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_ids[version],
        scheduler=load_scheduler(version, model_ids, scheduler),
        **kwargs,
    )
    pipe.safety_checker = None
    return pipe
