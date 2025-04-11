# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
from diffusers import StableDiffusionPipeline

from .utils import is_valid_version, load_scheduler

__all__ = [
    "_load_pipe",
]

model_ids = {
    "2-base": "stabilityai/stable-diffusion-2-base",
    "2": "stabilityai/stable-diffusion-2",
    "2-depth": "stabilityai/stable-diffusion-2-depth",
    "2.1-base": "stabilityai/stable-diffusion-2-1-base",
    "2.1": "stabilityai/stable-diffusion-2-1",
    "2.1-inpainting": "stabilityai/stable-diffusion-2-1-inpainting",
    "2.1-unclip": "stabilityai/stable-diffusion-2-1-unclip",
    "2.1-unclip-small": "stabilityai/stable-diffusion-2-1-unclip-small",
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
