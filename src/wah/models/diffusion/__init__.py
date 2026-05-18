from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch
import tqdm
from PIL.Image import Image
from torch import Tensor

from .pipe import load_pipe

__all__ = [
    "DiffusionModel",
]


@dataclass
class DiffusionOutput:
    images: List[Image]
    latents: List[Tensor]
    noise_preds: List[Tuple[Tensor, Tensor]]


class DiffusionModel:
    def __init__(
        self,
        name: str,
        scheduler: str = None,
        torch_dtype=torch.float32,
        use_safetensors: bool = True,
        compile_pipe: Optional[Literal["reduce-overhead", "max-autotune"]] = None,
        **kwargs,
    ):
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
            self.init(
                num_inference_steps=1,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
            _ = self.__call__("")
            self.reset()
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
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        num_images_per_prompt: int = None,
        **kwargs,
    ) -> None:
        if height is not None:
            kwargs["height"] = height
        if width is not None:
            kwargs["width"] = width
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if num_images_per_prompt is not None:
            kwargs["num_images_per_prompt"] = num_images_per_prompt

        self._utils._init(
            pipe=self._pipe,
            **kwargs,
        )

    def reset(self) -> None:
        self._utils._reset(
            pipe=self._pipe,
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> Tensor:
        return self._utils._encode_prompt(
            pipe=self._pipe,
            prompt=prompt,
            **kwargs,
        )

    @torch.no_grad()
    def prepare_prompt_embeds(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        return self._utils._prepare_prompt_embeds(
            pipe=self._pipe,
            prompt=prompt,
            **kwargs,
        )

    @torch.no_grad()
    def encode_image(
        self,
        image,
        **kwargs,
    ) -> Tensor:
        return self._utils._encode_image(
            pipe=self._pipe,
            image=image,
            **kwargs,
        )

    @torch.no_grad()
    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image,
        **kwargs,
    ) -> Tensor:
        return self._utils._prepare_ip_adapter_image_embeds(
            pipe=self._pipe,
            ip_adapter_image=ip_adapter_image,
            **kwargs,
        )

    @torch.no_grad()
    def prepare_latents(
        self,
        seed: int | list[int] = None,
        **kwargs,
    ) -> Tensor:
        return self._utils._prepare_latents(
            pipe=self._pipe,
            seed=seed,
            **kwargs,
        )

    def prepare_timesteps(
        self,
        **kwargs,
    ) -> Tuple[Tensor, int, int]:
        return self._utils._prepare_timesteps(
            pipe=self._pipe,
            **kwargs,
        )

    def predict_noise(
        self,
        latents: Tensor,
        t: Tensor,
        prompt_embeds: Union[Tensor, Tuple[Tensor, ...]],
        enable_grad: bool = False,
        **kwargs,
    ) -> Tensor:
        flag = torch.enable_grad() if enable_grad else torch.no_grad()
        with flag:
            return self._utils._predict_noise(
                pipe=self._pipe,
                latents=latents,
                t=t,
                prompt_embeds=prompt_embeds,
                **kwargs,
            )

    def perform_guidance(
        self,
        noise_preds: Union[Tensor, Tuple[Tensor, ...]],
        **kwargs,
    ) -> Tensor:
        return self._utils._perform_guidance(
            pipe=self._pipe,
            noise_preds=noise_preds,
            **kwargs,
        )

    def denoise_single_step(
        self,
        noise_preds: Tensor,
        t: Tensor,
        latents: Tensor,
        **kwargs,
    ) -> Tensor:
        return self._utils._denoise_single_step(
            pipe=self._pipe,
            noise_preds=noise_preds,
            t=t,
            latents=latents,
            **kwargs,
        )

    @torch.no_grad()
    def decode_latents(
        self,
        latents: Tensor,
        **kwargs,
    ) -> List[Image]:
        return self._utils._decode_latents(
            pipe=self._pipe,
            latents=latents,
            **kwargs,
        )

    def rearrange_latents(
        self,
        latents: List[Tensor],
    ) -> List[Tensor]:
        return self._utils._rearrange_latents(
            pipe=self._pipe,
            latents=latents,
        )

    def rearrange_noise_preds(
        self,
        noise_preds: List[Tensor],
    ) -> List[Union[Tensor, Tuple[Tensor, ...]]]:
        return self._utils._rearrange_noise_preds(
            pipe=self._pipe,
            noise_preds=noise_preds,
        )

    def __call__(
        self,
        prompt: str | list[str],
        seed: int | list[int] = None,
        verbose: bool = False,
        enable_grad: bool = False,
        **kwargs,
    ) -> List[Image]:
        prompt_embeds = self.prepare_prompt_embeds(
            prompt=prompt,
            **kwargs,
        )
        latents = self.prepare_latents(
            seed=seed,
            **kwargs,
        )
        timesteps, _, _ = self.prepare_timesteps(**kwargs)

        latents_list = [latents]
        noise_preds_list = []
        for t in tqdm.tqdm(timesteps, disable=not verbose):
            noise_preds = self.predict_noise(
                latents=latents,
                t=t,
                prompt_embeds=prompt_embeds,
                enable_grad=enable_grad,
                **kwargs,
            )
            noise_preds_list.append(noise_preds)
            noise_preds = self.perform_guidance(
                noise_preds=noise_preds,
                **kwargs,
            )
            latents = self.denoise_single_step(
                noise_preds=noise_preds,
                t=t,
                latents=latents,
                **kwargs,
            )
            latents_list.append(latents)
        images = self.decode_latents(
            latents=latents,
            **kwargs,
        )

        latents_list = self.rearrange_latents(latents_list)
        noise_preds_list = self.rearrange_noise_preds(noise_preds_list)

        return DiffusionOutput(
            images=images,
            latents=latents_list,
            noise_preds=noise_preds_list,
        )
