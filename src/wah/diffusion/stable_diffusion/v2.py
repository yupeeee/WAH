# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
import torch.nn as nn
from diffusers import StableDiffusionPipeline

from ...misc.typing import Device, List, Tensor
from ..utils import is_valid_version, load_scheduler
from .safety_checker import SafetyChecker
from .v1 import SDv1, SDv1Config

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


class SDv2Config(SDv1Config):
    pass


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


class SDv2(SDv1):
    def __init__(
        self,
        version: str,
        scheduler: str,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        self.config: SDv2Config = SDv2Config(pipe_id=f"SDv{version}_{scheduler}")

        self.pipe: StableDiffusionPipeline = _load_pipe(
            version=version,
            scheduler=scheduler,
            **kwargs,
        )
        self.device: Device = self.pipe._execution_device
        self.safety_checker = SafetyChecker(self.device)

        self.verbose: bool = verbose
        self.desc = lambda prompt: f"\033[1m{prompt}\033[0m@{self.config.pipe_id}"

        self._seed: int = None
        self._latents: List[Tensor] = []
        self._noise_preds: List[Tensor] = []
