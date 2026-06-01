from dataclasses import dataclass
from typing import Literal, Optional

import torch
import tqdm
from PIL import Image

from .pipe import load_pipe
from .pipelines.utils import InitState

__all__ = [
    "DiffusionModel",
]


@dataclass
class DiffusionOutput:
    images: list[Image.Image]
    latents: list[torch.Tensor]
    noise_preds: list[tuple[torch.Tensor, torch.Tensor]]


class DiffusionModel:
    def __init__(
        self,
        name: str,
        scheduler: str = None,
        torch_dtype=torch.float32,
        use_safetensors: bool = True,
        compile_pipe: Optional[Literal["reduce-overhead", "max-autotune"]] = None,
        **kwargs,
    ) -> None:
        self._pipe, self._utils = load_pipe(
            name=name,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            compile_pipe=compile_pipe,
            **kwargs,
        )

        if compile_pipe is not None:
            self.to("cuda" if torch.cuda.is_available() else "cpu")
            _ = self.__call__(
                prompt="",
                num_inference_steps=1,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
            # self.to("cpu") <- Removed due to the following reason:
            # Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device.
            # It is not recommended to move them to `cpu` as running them will fail.
            # Please make sure to use an accelerator to run the pipeline in inference,
            # due to the lack of support for`float16` operations on this device in PyTorch.
            # Please, remove the `torch_dtype=torch.float16` argument, or use another device for inference.

    def to(
        self,
        device: str | torch.device,
    ) -> "DiffusionModel":
        self._pipe.to(device)
        return self

    def init(
        self,
        prompt: str | list[str],
        seed: int | list[int] = None,
        **kwargs,
    ) -> InitState:
        return self._utils._init(
            pipe=self._pipe,
            prompt=prompt,
            seed=seed,
            **kwargs,
        )

    def reset(self) -> None:
        self._utils._reset(
            pipe=self._pipe,
        )

    def predict_noise(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        enable_grad: bool = False,
    ) -> torch.Tensor:
        flag = torch.enable_grad() if enable_grad else torch.no_grad()
        with flag:
            return self._utils._predict_noise(
                pipe=self._pipe,
                latents=latents,
                t=t,
            )

    def perform_guidance(
        self,
        noise_preds: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self._utils._perform_guidance(
            pipe=self._pipe,
            noise_preds=noise_preds,
        )

    def denoise_single_step(
        self,
        noise_preds: torch.Tensor,
        t: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return self._utils._denoise_single_step(
            pipe=self._pipe,
            noise_preds=noise_preds,
            t=t,
            latents=latents,
        )

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> list[Image.Image]:
        return self._utils._decode_latents(
            pipe=self._pipe,
            latents=latents,
        )

    def rearrange_latents(
        self,
        latents: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return self._utils._rearrange_latents(
            pipe=self._pipe,
            latents=latents,
        )

    def rearrange_noise_preds(
        self,
        noise_preds: list[torch.Tensor],
    ) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
        return self._utils._rearrange_noise_preds(
            pipe=self._pipe,
            noise_preds=noise_preds,
        )

    def __call__(
        self,
        prompt: str | list[str],
        seed: int | list[int] = None,
        verbose: bool = False,
        desc: str = None,
        enable_grad: bool = False,
        **kwargs,
    ) -> DiffusionOutput:
        init_state = self.init(
            prompt=prompt,
            seed=seed,
            **kwargs,
        )
        latents = init_state.latents
        timesteps = init_state.timesteps

        latents_list = [latents]
        noise_preds_list = []
        for t in tqdm.tqdm(timesteps, disable=not verbose, desc=desc):
            noise_preds = self.predict_noise(
                latents=latents,
                t=t,
                enable_grad=enable_grad,
            )
            noise_preds_list.append(noise_preds)
            noise_preds = self.perform_guidance(
                noise_preds=noise_preds,
            )
            latents = self.denoise_single_step(
                noise_preds=noise_preds,
                t=t,
                latents=latents,
            )
            latents_list.append(latents)
        images = self.decode_latents(
            latents=latents,
        )

        latents_list = self.rearrange_latents(latents_list)
        noise_preds_list = self.rearrange_noise_preds(noise_preds_list)

        self.reset()

        return DiffusionOutput(
            images=images,
            latents=latents_list,
            noise_preds=noise_preds_list,
        )
