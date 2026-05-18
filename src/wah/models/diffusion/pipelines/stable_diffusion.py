from typing import Any, List, Tuple, Union

import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from PIL.Image import Image
from torch import Tensor

from .utils import (
    _get_generator,
    _get_image_embeds,
    _get_prompt_embeds,
    _rescale_noise_cfg,
    _retrieve_timesteps,
)

__all__ = [
    "_Pipeline",
    "_NoisePredictorName",
    "_init",
    "_reset",
    "_encode_prompt",
    "_prepare_prompt_embeds",
    "_encode_image",
    "_prepare_ip_adapter_image_embeds",
    "_prepare_latents",
    "_prepare_timesteps",
    "_predict_noise",
    "_perform_guidance",
    "_denoise_single_step",
    "_decode_latents",
    "_rearrange_latents",
    "_rearrange_noise_preds",
]

_Pipeline = StableDiffusionPipeline
_NoisePredictorName = "unet"


def _init(
    pipe: _Pipeline,
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = 50,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    num_images_per_prompt: int = 1,
    clip_skip: int | None = None,
    cross_attention_kwargs: dict[str, Any] | None = None,
) -> _Pipeline:
    # 0. Default height and width to unet
    if not height or not width:
        height = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[0]
        )
        width = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[1]
        )
        height, width = height * pipe.vae_scale_factor, width * pipe.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
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

    pipe._guidance_scale = guidance_scale
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._interrupt = False

    pipe._params = {
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "timesteps": timesteps,
        "sigmas": sigmas,
        "batch_size": None,  # to be updated at _encode_prompt
        "num_images_per_prompt": num_images_per_prompt,
        "lora_scale": (
            pipe.cross_attention_kwargs.get("scale", None)
            if pipe.cross_attention_kwargs is not None
            else None
        ),
        "added_cond_kwargs": None,
        "timestep_cond": None,
        "device": pipe._execution_device,
        "dtype": pipe.unet.dtype,
    }

    return pipe


def _reset(pipe: _Pipeline) -> _Pipeline:
    pipe._guidance_scale = 7.5
    pipe._guidance_rescale = 0.0
    pipe._clip_skip = None
    pipe._cross_attention_kwargs = None
    pipe._interrupt = False

    pipe._params = {
        "height": None,
        "width": None,
        "num_inference_steps": 50,
        "timesteps": None,
        "sigmas": None,
        "batch_size": None,
        "num_images_per_prompt": 1,
        "lora_scale": None,
        "added_cond_kwargs": None,
        "timestep_cond": None,
        "device": pipe._execution_device,
        "dtype": pipe.unet.dtype,
    }

    return pipe


def _encode_prompt(
    pipe: _Pipeline,
    prompt: str | list[str],
) -> Tensor:
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    lora_scale = pipe._params["lora_scale"]
    device = pipe._params["device"]
    dtype = pipe._params["dtype"]

    prompt_embeds = _get_prompt_embeds(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        prompt=prompt,
        max_sequence_length=pipe.tokenizer.model_max_length,
        use_pooled_output=False,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        dtype=dtype,
        lora_scale=lora_scale,
        clip_skip=pipe.clip_skip,
    )

    pipe._params["batch_size"] = prompt_embeds.shape[0]

    return prompt_embeds


def _prepare_prompt_embeds(
    pipe: _Pipeline,
    prompt: str | list[str],
    negative_prompt: str | list[str] = None,
) -> Tensor:
    prompt_embeds = _encode_prompt(
        pipe=pipe,
        prompt=prompt,
    )

    if pipe.do_classifier_free_guidance:
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        negative_prompt_embeds = _encode_prompt(
            pipe=pipe,
            prompt=uncond_tokens,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds


def _encode_image(
    pipe: _Pipeline,
    image,
) -> Tensor:
    num_images_per_prompt = pipe._params["num_images_per_prompt"]

    image_embeds = _get_image_embeds(
        pipe=pipe,
        image=image,
        num_images_per_prompt=num_images_per_prompt,
    )

    return image_embeds


def _prepare_ip_adapter_image_embeds(
    pipe: _Pipeline,
    ip_adapter_image,
) -> Tensor:
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    device = pipe._params["device"]
    num_ip_adapters = len(pipe.unet.encoder_hid_proj.image_projection_layers)

    image_embeds = []
    if pipe.do_classifier_free_guidance:
        negative_image_embeds = []

    if not isinstance(ip_adapter_image, list):
        ip_adapter_image = [ip_adapter_image]

    if len(ip_adapter_image) != num_ip_adapters:
        raise ValueError(
            f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {num_ip_adapters} IP Adapters."
        )

    for single_ip_adapter_image in ip_adapter_image:
        single_image_embeds = _encode_image(pipe, single_ip_adapter_image, 1)
        image_embeds.append(single_image_embeds[None, :])
        if pipe.do_classifier_free_guidance:
            negative_single_image_embeds = torch.zeros_like(image_embeds)
            negative_image_embeds.append(negative_single_image_embeds[None, :])

    ip_adapter_image_embeds = []
    for i in range(num_ip_adapters):
        single_image_embeds = torch.cat(
            [image_embeds[i]] * num_images_per_prompt, dim=0
        )
        if pipe.do_classifier_free_guidance:
            single_negative_image_embeds = torch.cat(
                [negative_image_embeds[i]] * num_images_per_prompt, dim=0
            )
            single_image_embeds = torch.cat(
                [single_negative_image_embeds, single_image_embeds], dim=0
            )

        single_image_embeds = single_image_embeds.to(device=device)
        ip_adapter_image_embeds.append(single_image_embeds)

    # 6.1 Add image embeds for IP-Adapter
    pipe._params["added_cond_kwargs"] = {"image_embeds": ip_adapter_image_embeds}

    return ip_adapter_image_embeds


def _prepare_latents(
    pipe: _Pipeline,
    seed: int | list[int] | None = None,
) -> Tensor:
    batch_size = pipe._params["batch_size"]
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    num_channels_latents = pipe.unet.config.in_channels
    height = pipe._params["height"]
    width = pipe._params["width"]
    dtype = pipe._params["dtype"]
    device = pipe._params["device"]
    generator = _get_generator(seed, batch_size * num_images_per_prompt, device)

    latents = pipe.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    return latents


def _prepare_timesteps(
    pipe: _Pipeline,
) -> Tuple[Tensor, int, int]:
    num_inference_steps = pipe._params["num_inference_steps"]
    batch_size = pipe._params["batch_size"]
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    timesteps = pipe._params["timesteps"]
    sigmas = pipe._params["sigmas"]
    device = pipe._params["device"]
    dtype = pipe._params["dtype"]

    timesteps = _retrieve_timesteps(
        scheduler=pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )
    num_inference_steps = len(timesteps)
    pipe._num_inference_steps = num_inference_steps
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * pipe.scheduler.order, 0
    )

    # 6.2 Optionally get Guidance Scale Embedding
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=dtype)

        pipe._params["timestep_cond"] = timestep_cond

    return timesteps, num_inference_steps, num_warmup_steps


def _predict_noise(
    pipe: _Pipeline,
    latents: Tensor,
    t: Tensor,
    prompt_embeds: Tensor,
) -> Tensor:
    if pipe.interrupt:
        return None

    # expand the latents if we are doing classifier free guidance
    latents = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
    if hasattr(pipe.scheduler, "scale_model_input"):
        latents = pipe.scheduler.scale_model_input(latents, t)

    # predict the noise residual
    noise_preds = pipe.unet(
        latents,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=pipe._params["timestep_cond"],
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs=pipe._params["added_cond_kwargs"],
        return_dict=False,
    )[0]

    return noise_preds


def _perform_guidance(
    pipe: _Pipeline,
    noise_preds: Tensor,
) -> Tensor:
    if pipe.do_classifier_free_guidance:
        noise_preds_uncond, noise_preds_cond = noise_preds.chunk(2)
        noise_preds = noise_preds_uncond + pipe.guidance_scale * (
            noise_preds_cond - noise_preds_uncond
        )

    if pipe.do_classifier_free_guidance and pipe.guidance_rescale > 0.0:
        # Based on 3.4. in https://huggingface.co/papers/2305.08891
        noise_preds = _rescale_noise_cfg(
            noise_preds, noise_preds_cond, guidance_rescale=pipe.guidance_rescale
        )

    return noise_preds


def _denoise_single_step(
    pipe: _Pipeline,
    noise_preds: Tensor,
    t: Tensor,
    latents: Tensor,
) -> Tensor:
    latents = pipe.scheduler.step(noise_preds, t, latents, return_dict=False)[0]

    return latents


def _decode_latents(
    pipe: _Pipeline,
    latents: Tensor,
) -> List[Image]:
    images = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor,
        return_dict=False,
        generator=pipe._params.get("generator"),
    )[0]
    images = pipe.image_processor.postprocess(
        images, output_type="pil", do_denormalize=[True] * images.shape[0]
    )

    return images


def _rearrange_latents(
    pipe: _Pipeline,
    latents: List[Tensor],  # List[Tensor(batch_size, C, H, W) * num_inference_steps]
) -> List[Tensor]:
    latents = torch.stack(
        latents, dim=1
    ).cpu()  #  Tensor(batch_size, num_inference_steps, C, H, W)
    latents = [
        latent for latent in latents
    ]  # List[Tensor(num_inference_steps, C, H, W) * batch_size]

    return latents


def _rearrange_noise_preds(
    pipe: _Pipeline,
    noise_preds: List[
        Tensor
    ],  # List[Tensor(batch_size * 2, C, H, W) * num_inference_steps]
) -> List[Union[Tensor, Tuple[Tensor, Tensor]]]:
    noise_preds = torch.stack(
        noise_preds, dim=1
    ).cpu()  # Tensor(batch_size(* 2), num_inference_steps, C, H, W)
    if pipe.do_classifier_free_guidance:
        noise_preds_uncond, noise_preds_cond = noise_preds.chunk(
            2, dim=0
        )  # Tensor(batch_size, num_inference_steps, C, H, W) * 2
        noise_preds = [
            (noise_pred_uncond, noise_pred_cond)
            for noise_pred_uncond, noise_pred_cond in zip(
                noise_preds_uncond, noise_preds_cond
            )
        ]  # List[Tuple[Tensor(num_inference_steps, C, H, W) * 2] * batch_size]
    else:
        noise_preds = [
            noise_pred for noise_pred in noise_preds
        ]  # List[Tensor(num_inference_steps, C, H, W) * batch_size]

    return noise_preds
