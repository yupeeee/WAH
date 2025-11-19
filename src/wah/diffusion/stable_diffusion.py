import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import diffusers
import torch
import torch.nn as nn
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.utils import is_torch_xla_available
from PIL.Image import Image
from transformers import CLIPConfig, CLIPVisionModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

__all__ = [
    "StableDiffusion",
]


SUPPORTED_VERSIONS = {
    "1.1": "CompVis/stable-diffusion-v1-1",
    "1.2": "CompVis/stable-diffusion-v1-2",
    "1.3": "CompVis/stable-diffusion-v1-3",
    "1.4": "CompVis/stable-diffusion-v1-4",
    "1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "2-base": "stabilityai/stable-diffusion-2-base",
    "2": "stabilityai/stable-diffusion-2",
    "2.1-base": "stabilityai/stable-diffusion-2-1-base",
    "2.1": "stabilityai/stable-diffusion-2-1",
}


def _load_scheduler(
    version: str,
    strategy: str,
    **kwargs,
) -> diffusers.SchedulerMixin:
    try:
        scheduler = getattr(diffusers, f"{strategy}Scheduler").from_pretrained(
            pretrained_model_name_or_path=SUPPORTED_VERSIONS[version],
            subfolder="scheduler",
            **kwargs,
        )
        return scheduler
    except AttributeError as e:
        raise ValueError(f"Unknown scheduler: {strategy}Scheduler ({e})")


def _load_pipe(
    version: str,
    scheduler: str,
    supported_versions: Dict[str, str],
    variant: str = None,
    verbose: bool = False,
    safety_check: bool = True,
    **kwargs,
) -> diffusers.StableDiffusionPipeline:
    assert version in supported_versions, f"Unsupported version: {version}"

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=supported_versions[version],
        scheduler=_load_scheduler(version, scheduler),
        variant=variant,
        **kwargs,
    )

    if not verbose:
        pipe.set_progress_bar_config(disable=True)

    if not safety_check:
        pipe.safety_checker = None

    return pipe


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion:
    """
    [Stable Diffusion](https://arxiv.org/abs/2112.10752) Pipeline for text-to-image generation.

    ## Example (Pipeline call)
    ```python
    import wah

    # Load pipeline
    pipe = wah.diffusion.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
    )
    pipe.to("cuda")
    pipe.verbose(True)

    # Denoising
    prompt = "a photo of an astronaut riding a horse on mars"
    images = pipe(prompt).images
    images[0].save("image.png")
    ```

    ## Example (Manual denoising)
    ```python
    import wah

    # Load pipeline
    pipe = wah.diffusion.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
    )

    # Initialize
    pipe.to("cuda")
    pipe.init(
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # Prepare denoising
    prompt = "a photo of an astronaut riding a horse on mars"
    prompt_embeds = pipe.prepare_embeds(prompt)
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=0)

    # Denoising loop
    for i, t in enumerate(timesteps):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)
        noise_pred = pipe.perform_guidance(noise_pred)
        latents = pipe.step(noise_pred, t, latents)

    # Decode image
    images = pipe.decode(latents)
    images[0].save("image.png")
    ```
    """

    _SUPPORTED_VERSIONS = SUPPORTED_VERSIONS

    def __init__(
        self,
        version: str,
        scheduler: str,
        variant: str = None,
        verbose: bool = False,
        safety_check: bool = True,
        **kwargs,
    ) -> None:
        self.pipe: diffusers.StableDiffusionPipeline = _load_pipe(
            version,
            scheduler,
            supported_versions=self._SUPPORTED_VERSIONS,
            variant=variant,
            verbose=verbose,
            safety_check=safety_check,
            **kwargs,
        )
        self.device: torch.device = self.pipe._execution_device

        # encode_image
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L41
        config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )

    @property
    def supported_versions(self) -> List[str]:
        return list(self._SUPPORTED_VERSIONS.keys())

    def to(
        self,
        device: Optional[Union[str, torch.device]],
    ) -> "StableDiffusion":
        self.pipe.to(device)
        self.device = device

        return self

    def verbose(
        self,
        _verbose: bool,
    ) -> "StableDiffusion":
        self.pipe.set_progress_bar_config(disable=not _verbose)

        return self

    def alphas_cumprod(
        self,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        alphas_cumprod = []
        for step in range(0, num_inference_steps):
            t = self.pipe.scheduler.timesteps[step]
            alphas_cumprod.append(self.pipe.scheduler.alphas_cumprod[t.item()])

        return torch.stack(alphas_cumprod, dim=0)  # (num_inference_steps, )

    def reset(self) -> "StableDiffusion":
        self.pipe._guidance_scale = 7.5
        self.pipe._guidance_rescale = 0.0
        self.pipe._clip_skip = None
        self.pipe._cross_attention_kwargs = None
        self.pipe._interrupt = True

        # init
        self._height = None
        self._width = None
        self._num_inference_steps = 50
        self._timesteps = None
        self._sigmas = None
        self._eta = 0.0

        # prepare_embeds
        self._added_cond_kwargs = None

        # prepare_latents
        self._generator = None
        self._extra_step_kwargs = None
        self._timestep_cond = None

        return self

    def init(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
    ) -> "StableDiffusion":
        self.reset()

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.pipe.unet.config.sample_size
                if self.pipe._is_unet_config_sample_size_int
                else self.pipe.unet.config.sample_size[0]
            )
            width = (
                self.pipe.unet.config.sample_size
                if self.pipe._is_unet_config_sample_size_int
                else self.pipe.unet.config.sample_size[1]
            )
            height, width = (
                height * self.pipe.vae_scale_factor,
                width * self.pipe.vae_scale_factor,
            )
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(
            prompt="",  # dummy prompt
            height=height,
            width=width,
            callback_steps=None,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
        )

        self.pipe._guidance_scale = guidance_scale
        self.pipe._guidance_rescale = guidance_rescale
        self.pipe._clip_skip = clip_skip
        self.pipe._cross_attention_kwargs = cross_attention_kwargs
        self.pipe._interrupt = False

        self._height = height
        self._width = width
        self._num_inference_steps = num_inference_steps
        self._timesteps = timesteps
        self._sigmas = sigmas
        self._num_images_per_prompt = num_images_per_prompt
        self._eta = eta

        return self

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lora_scale = (
            self.pipe.cross_attention_kwargs.get("scale", None)
            if self.pipe.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            self.pipe.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.pipe.clip_skip,
        )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image, List[Image]],
    ) -> torch.Tensor:
        clip_input = self.pipe.feature_extractor(
            image, return_tensors="pt"
        ).pixel_values
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L52
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        return image_embeds

    def encode(
        self,
        images: List[Image],
    ) -> torch.Tensor:
        if isinstance(images, Image):
            images = [images]
        images = self.pipe.image_processor.preprocess(
            image=images,
            height=self.pipe.unet.config.sample_size * 8,
            width=self.pipe.unet.config.sample_size * 8,
        ).to(self.device)
        latents = self.pipe.vae.encode(images).latent_dist.mode()
        latents = latents * self.pipe.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
    ) -> List[Image]:
        decoded = self.pipe.vae.decode(
            latents.to(self.device) / self.pipe.vae.config.scaling_factor,
            return_dict=False,
            generator=self._generator,
        )[0]
        pil_images = self.pipe.image_processor.postprocess(
            decoded,
            output_type="pil",
            do_denormalize=[True] * decoded.shape[0],
        )

        return pil_images

    def prepare_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self._num_images_per_prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                batch_size * self._num_images_per_prompt,
                self.pipe.do_classifier_free_guidance,
            )

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        self._added_cond_kwargs = added_cond_kwargs

        return prompt_embeds

    def prepare_timesteps(
        self,
    ) -> Tuple[List[int], int, int]:
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            self._num_inference_steps,
            self.device,
            self._timesteps,
            self._sigmas,
        )

        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.pipe.scheduler.order
        )

        return timesteps, num_inference_steps, num_warmup_steps

    def prepare_latents(
        self,
        prompt_embeds: torch.Tensor,
        seed: Optional[Union[int, List[int]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = len(prompt_embeds)
        if self.pipe.do_classifier_free_guidance:
            batch_size = batch_size // 2

        if seed is None:
            generator = None
        else:
            if isinstance(seed, int):
                seed = [seed] * batch_size
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = [
                torch.Generator(device=self.device).manual_seed(s) for s in seed
            ]
        self._generator = generator

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * self._num_images_per_prompt,
            num_channels_latents,
            self._height,
            self._width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, self._eta)
        self._extra_step_kwargs = extra_step_kwargs

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(
                batch_size * self._num_images_per_prompt
            )
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.pipe.unet.config.time_cond_proj_dim,
            ).to(device=self.device, dtype=latents.dtype)
        self._timestep_cond = timestep_cond

        return latents

    @torch.no_grad()
    def predict_noise(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if self.pipe.do_classifier_free_guidance
            else latents
        )
        if hasattr(self.pipe.scheduler, "scale_model_input"):
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )

        # predict the noise residual
        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=self._timestep_cond,
            cross_attention_kwargs=self.pipe._cross_attention_kwargs,
            added_cond_kwargs=self._added_cond_kwargs,
            return_dict=False,
        )[0]

        return noise_pred

    def perform_guidance(
        self,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        # perform guidance
        if self.pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.pipe.do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
            # Based on 3.4. in https://huggingface.co/papers/2305.08891
            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_text,
                guidance_rescale=self.pipe.guidance_rescale,
            )

        return noise_pred

    def step(
        self,
        noise_pred: torch.Tensor,
        t: int,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.pipe.scheduler.step(
            noise_pred, t, latents, **self._extra_step_kwargs, return_dict=False
        )[0]

        return latents

    def generate(
        self,
        prompt: Union[str, List[str]] = None,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[List[Image], List[torch.Tensor], List[torch.Tensor]]:
        prompt_embeds = self.prepare_embeds(prompt)
        timesteps, _, _ = self.prepare_timesteps()
        x_t = self.prepare_latents(prompt_embeds, seed)

        latents = [x_t.cpu()]
        noise_preds = []

        for i, t in enumerate(timesteps):
            noise_pred = self.predict_noise(x_t, prompt_embeds, t)
            noise_preds.append(noise_pred.cpu())
            noise_pred = self.perform_guidance(noise_pred)
            x_t = self.step(noise_pred, t, x_t)
            latents.append(x_t.cpu())

        # Rearrange latents and noise predictions
        latents = [latent for latent in torch.stack(latents, dim=0).transpose(0, 1)]
        if self.pipe.do_classifier_free_guidance:
            eps_ts_uncond, eps_ts_cond = (
                torch.stack(noise_preds, dim=0).transpose(0, 1).chunk(2, dim=0)
            )
            noise_preds = [
                (eps_t_uncond, eps_t_cond)
                for eps_t_uncond, eps_t_cond in zip(eps_ts_uncond, eps_ts_cond)
            ]
        else:
            noise_preds = [noise_pred for noise_pred in torch.stack(noise_preds, dim=0).transpose(0, 1)]

        # Decode latents
        x_0 = torch.stack([latent[-1] for latent in latents], dim=0)
        images = self.decode(x_0)

        return images, latents, noise_preds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        seed: Optional[Union[int, List[int]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        # initialize pipeline
        self.init(
            height,
            width,
            num_inference_steps,
            timesteps,
            sigmas,
            guidance_scale,
            num_images_per_prompt,
            eta,
            cross_attention_kwargs,
            guidance_rescale,
            clip_skip,
            **kwargs,
        )

        # prepare embeds
        prompt_embeds = self.prepare_embeds(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
        )

        # prepare timesteps
        timesteps, num_inference_steps, num_warmup_steps = self.prepare_timesteps()

        # prepare latents
        latents = self.prepare_latents(
            prompt_embeds,
            seed,
            latents,
        )

        # 7. Denoising loop
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue

                noise_pred = self.predict_noise(latents, prompt_embeds, t)

                noise_pred = self.perform_guidance(noise_pred)

                latents = self.step(noise_pred, t, latents)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self.pipe, i, t, callback_kwargs
                    )

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.pipe.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
                generator=self._generator,
            )[0]
            if self.pipe.safety_checker is not None:
                image, has_nsfw_concept = self.pipe.run_safety_checker(
                    image, self.device, prompt_embeds.dtype
                )
            else:
                has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.pipe.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
