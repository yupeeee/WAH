from diffusers import StableDiffusionPipeline

from ..utils import is_valid_version
from .scheduler import load_scheduler

__all__ = [
    "load_pipeline",
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


def load_pipeline(
    version: str,
    scheduler: str,
    **kwargs,
) -> StableDiffusionPipeline:
    assert is_valid_version(version, model_ids)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_ids[version],
        scheduler=load_scheduler(version, model_ids, scheduler),
        **kwargs,
    )
    pipeline.safety_checker = None
    return pipeline
