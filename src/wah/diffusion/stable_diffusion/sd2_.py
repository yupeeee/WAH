# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
import torch.nn as nn
from diffusers import StableDiffusionPipeline

from ...misc.typing import Any, Dict, List, Tensor
from ..utils import is_valid_version
from .safety_checker import SafetyChecker
from .scheduler import load_scheduler
from .sd1_ import SDv1, SDv1Config, UNet

__all__ = [
    "SDv2",
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


class SDv2Config(SDv1Config):
    pass


class SDv2(SDv1):
    def __init__(
        self,
        version: str,
        scheduler: str,
        blur_nsfw: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        self.config = SDv2Config()

        self.pipe = load_pipeline(version, scheduler, **kwargs)
        self.device = self.pipe._execution_device
        self._unet = UNet(self.pipe, None, self.config.guidance_scale).to(self.device)

        self.blur_nsfw = blur_nsfw
        self.safety_checker = SafetyChecker(self.device)

        self.verbose = verbose
        self.desc = lambda prompt: f"\033[1m{prompt}\033[0m@SDv{version}_{scheduler}"

        self._seed: int = None
        self._timesteps: Tensor = self.config.timesteps
        self._num_inference_steps: int = self.config.num_inference_steps
        self._unet_args: Dict[str, Any] = {}
        self._params: Dict[str, Any] = {}

        self.noise_preds: List[Tensor] = []
        self.latents: List[Tensor] = []
