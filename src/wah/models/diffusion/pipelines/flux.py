from typing import Any, List, Tuple, Union

import numpy as np
import torch
from diffusers.pipelines.flux import FluxPipeline
from PIL.Image import Image
from torch import Tensor

from .utils import (
    _calculate_shift,
    _detach_clone,
    _get_generator,
    _get_image_embeds,
    _get_prompt_embeds,
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

_Pipeline = FluxPipeline
_NoisePredictorName = "transformer"


def _init(
    pipe: FluxPipeline,
    true_cfg_scale: float = 1.0,
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = 28,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: int = 1,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> FluxPipeline:
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
        prompt="",  # dummy prompt
        prompt_2=None,
        height=height,
        width=width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    pipe._guidance_scale = guidance_scale
    pipe._joint_attention_kwargs = joint_attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    pipe._params = {
        "true_cfg_scale": true_cfg_scale,
        "do_true_cfg": None,  # to be updated at _prepare_prompt_embeds
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "timesteps": timesteps,
        "sigmas": sigmas,
        "batch_size": None,  # to be updated at _encode_prompt
        "num_images_per_prompt": num_images_per_prompt,
        "lora_scale": (
            pipe.joint_attention_kwargs.get("scale", None)
            if pipe.joint_attention_kwargs is not None
            else None
        ),
        "device": pipe._execution_device,
        "dtype": pipe.transformer.dtype,
        "ip_adapter_image_embeds": None,  # to be updated at _prepare_ip_adapter_image_embeds
        "negative_ip_adapter_image_embeds": None,  # to be updated at _prepare_ip_adapter_image_embeds
        "latent_image_ids": None,  # to be updated at _prepare_latents
        "image_seq_len": None,  # to be updated at _prepare_latents
        "guidance": None,  # to be updated at _prepare_latents
    }

    return pipe


def _reset(pipe: FluxPipeline) -> FluxPipeline:
    pipe._guidance_scale = 3.5
    pipe._joint_attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    pipe._params = {
        "true_cfg_scale": 1.0,
        "do_true_cfg": None,
        "height": None,
        "width": None,
        "num_inference_steps": 28,
        "timesteps": None,
        "sigmas": None,
        "batch_size": None,
        "num_images_per_prompt": 1,
        "lora_scale": None,
        "device": pipe._execution_device,
        "dtype": pipe.transformer.dtype,
        "ip_adapter_image_embeds": None,
        "negative_ip_adapter_image_embeds": None,
        "latent_image_ids": None,
        "image_seq_len": None,
        "guidance": None,
    }

    return pipe


def _encode_prompt(
    pipe: FluxPipeline,
    prompt: str | list[str],
    prompt_2: str | list[str] | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    max_sequence_length = 77  # Default is 512 in https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py but defined as 77 in https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    device = pipe._params["device"]
    dtype = pipe._params["dtype"]
    lora_scale = pipe._params["lora_scale"]

    pooled_prompt_embeds = _get_prompt_embeds(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        use_pooled_output=True,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        dtype=dtype,
        lora_scale=lora_scale,
        clip_skip=None,
    )

    prompt_2 = prompt_2 or prompt
    prompt_embeds = _get_prompt_embeds(
        tokenizer=pipe.tokenizer_2,
        text_encoder=pipe.text_encoder_2,
        prompt=prompt_2,
        max_sequence_length=max_sequence_length,
        use_pooled_output=False,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        dtype=dtype,
        lora_scale=lora_scale,
        clip_skip=None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    pipe._params["batch_size"] = prompt_embeds.shape[0]

    return prompt_embeds, pooled_prompt_embeds, text_ids


def _prepare_prompt_embeds(
    pipe: FluxPipeline,
    prompt: str | list[str],
    prompt_2: str | list[str] = None,
    negative_prompt: str | list[str] = None,
    negative_prompt_2: str | list[str] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    true_cfg_scale = pipe._params["true_cfg_scale"]

    has_neg_prompt = negative_prompt is not None
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    pipe._params["do_true_cfg"] = do_true_cfg

    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = _encode_prompt(
        pipe=pipe,
        prompt=prompt,
        prompt_2=prompt_2,
    )

    negative_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    negative_text_ids = None
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = _encode_prompt(
            pipe=pipe,
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
        )

    return (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        negative_text_ids,
    )


def _encode_image(
    pipe: FluxPipeline,
    image,
) -> Tensor:
    num_images_per_prompt = pipe._params["num_images_per_prompt"]

    image_embeds = _get_image_embeds(
        pipe=pipe,
        image=image,
        num_images_per_prompt=num_images_per_prompt,
    )

    return image_embeds


def __prepare_ip_adapter_image_embeds(
    pipe: FluxPipeline,
    ip_adapter_image,
) -> Tensor:
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    device = pipe._params["device"]
    num_ip_adapters = pipe.transformer.encoder_hid_proj.num_ip_adapters

    image_embeds = []

    if not isinstance(ip_adapter_image, list):
        ip_adapter_image = [ip_adapter_image]

    if len(ip_adapter_image) != num_ip_adapters:
        raise ValueError(
            f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {num_ip_adapters} IP Adapters."
        )

    for single_ip_adapter_image in ip_adapter_image:
        single_image_embeds = _encode_image(pipe, single_ip_adapter_image, 1)
        image_embeds.append(single_image_embeds[None, :])

    ip_adapter_image_embeds = []
    for i in range(num_ip_adapters):
        single_image_embeds = torch.cat(
            [image_embeds[i]] * num_images_per_prompt, dim=0
        )
        single_image_embeds = single_image_embeds.to(device=device)
        ip_adapter_image_embeds.append(single_image_embeds)

    return ip_adapter_image_embeds


def _prepare_ip_adapter_image_embeds(
    pipe: FluxPipeline,
    ip_adapter_image,
    negative_ip_adapter_image,
) -> Tuple[Tensor, Tensor]:
    height = pipe._params["height"]
    width = pipe._params["width"]
    num_ip_adapters = pipe.transformer.encoder_hid_proj.num_ip_adapters

    if (ip_adapter_image is not None) and (negative_ip_adapter_image is None):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        negative_ip_adapter_image = [negative_ip_adapter_image] * num_ip_adapters

    elif (ip_adapter_image is None) and (negative_ip_adapter_image is not None):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        ip_adapter_image = [ip_adapter_image] * num_ip_adapters

    if pipe._joint_attention_kwargs is None:
        pipe._joint_attention_kwargs = {}

    ip_adapter_image_embeds = None
    negative_ip_adapter_image_embeds = None
    if ip_adapter_image is not None:
        ip_adapter_image_embeds = __prepare_ip_adapter_image_embeds(
            pipe,
            ip_adapter_image,
        )
        pipe._params["ip_adapter_image_embeds"] = ip_adapter_image_embeds
    if negative_ip_adapter_image is not None:
        negative_ip_adapter_image_embeds = __prepare_ip_adapter_image_embeds(
            pipe,
            negative_ip_adapter_image,
        )
        pipe._params["negative_ip_adapter_image_embeds"] = (
            negative_ip_adapter_image_embeds
        )

    return ip_adapter_image_embeds, negative_ip_adapter_image_embeds


def _prepare_latents(
    pipe: FluxPipeline,
    seed: int | list[int] | None = None,
) -> Tensor:
    batch_size = pipe._params["batch_size"]
    num_images_per_prompt = pipe._params["num_images_per_prompt"]
    num_channels_latents = pipe.transformer.config.in_channels // 4
    height = pipe._params["height"]
    width = pipe._params["width"]
    dtype = pipe._params["dtype"]
    device = pipe._params["device"]
    generator = _get_generator(seed, batch_size * num_images_per_prompt, device)

    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    pipe._params["latent_image_ids"] = latent_image_ids
    pipe._params["image_seq_len"] = latents.shape[1]

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], pipe.guidance_scale, device=device, dtype=dtype)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None
    pipe._params["guidance"] = guidance

    if pipe._joint_attention_kwargs is None:
        pipe._joint_attention_kwargs = {}

    return latents


def _prepare_timesteps(
    pipe: FluxPipeline,
) -> Tuple[Tensor, int, int]:
    num_inference_steps = pipe._params["num_inference_steps"]
    sigmas = pipe._params["sigmas"]
    device = pipe._params["device"]
    timesteps = pipe._params["timesteps"]
    image_seq_len = pipe._params["image_seq_len"]

    sigmas = (
        np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if sigmas is None
        else sigmas
    )
    if (
        hasattr(pipe.scheduler.config, "use_flow_sigmas")
        and pipe.scheduler.config.use_flow_sigmas
    ):
        sigmas = None

    mu = _calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )

    timesteps = _retrieve_timesteps(
        scheduler=pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
        mu=mu,
    )
    num_inference_steps = len(timesteps)
    pipe._num_inference_steps = num_inference_steps
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * pipe.scheduler.order, 0
    )

    return timesteps, num_inference_steps, num_warmup_steps


def _predict_noise(
    pipe: FluxPipeline,
    latents: Tensor,
    t: Tensor,
    prompt_embeds: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> Tuple[Tensor, Tensor]:
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        negative_text_ids,
    ) = prompt_embeds

    if pipe.interrupt:
        return None

    pipe._current_timestep = t
    if pipe._params["ip_adapter_image_embeds"] is not None:
        pipe._joint_attention_kwargs["ip_adapter_image_embeds"] = pipe._params[
            "ip_adapter_image_embeds"
        ]
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    with pipe.transformer.cache_context("cond"):
        noise_preds = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=pipe._params["guidance"],
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=pipe._params["latent_image_ids"],
            joint_attention_kwargs=pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]
    noise_preds = _detach_clone(noise_preds)

    negative_noise_preds = None
    if pipe._params["do_true_cfg"]:
        if pipe._params["negative_image_embeds"] is not None:
            pipe._joint_attention_kwargs["ip_adapter_image_embeds"] = pipe._params[
                "negative_image_embeds"
            ]

        with pipe.transformer.cache_context("uncond"):
            negative_noise_preds = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=pipe._params["guidance"],
                pooled_projections=negative_pooled_prompt_embeds,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=pipe._params["latent_image_ids"],
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]

        if negative_noise_preds.requires_grad:
            raise RuntimeError("negative_noise_preds should not require gradients")

        negative_noise_preds = _detach_clone(negative_noise_preds)

    return noise_preds, negative_noise_preds


def _perform_guidance(
    pipe: _Pipeline,
    noise_preds: Tuple[Tensor, Tensor],
) -> Tensor:
    noise_preds, negative_noise_preds = noise_preds

    if pipe._params["do_true_cfg"]:
        noise_preds = negative_noise_preds + pipe._params["true_cfg_scale"] * (
            noise_preds - negative_noise_preds
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
    height = pipe._params["height"]
    width = pipe._params["width"]

    pipe._current_timestep = None

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")

    return images


def _rearrange_latents(
    pipe: _Pipeline,
    latents: List[Tensor],  # List[Tensor(batch_size, C, H, W) * num_inference_steps]
) -> List[Tensor]:
    latents = torch.stack(
        latents, dim=1
    ).cpu()  # Tensor(batch_size, num_inference_steps, C, H, W)
    latents = [
        latent for latent in latents
    ]  # List[Tensor(num_inference_steps, C, H, W) * batch_size]

    return latents


def _rearrange_noise_preds(
    pipe: _Pipeline,
    noise_preds: List[
        Union[Tensor, Tuple[Tensor, Tensor]]
    ],  # List[Tuple[Tensor(num_inference_steps, C, H, W) * 2] * batch_size]
) -> List[Union[Tensor, Tuple[Tensor, Tensor]]]:
    noise_preds = _rearrange_latents(
        pipe=pipe, latents=[noise_pred[0] for noise_pred in noise_preds]
    )  # List[Tensor(num_inference_steps, C, H, W) * batch_size]
    if pipe._params["do_true_cfg"]:
        negative_noise_preds = _rearrange_latents(
            pipe=pipe, latents=[noise_pred[1] for noise_pred in noise_preds]
        )  # List[Tensor(num_inference_steps, C, H, W) * batch_size]
        noise_preds = [
            (noise_pred, negative_noise_pred)
            for noise_pred, negative_noise_pred in zip(
                noise_preds, negative_noise_preds
            )
        ]  # List[Tuple[Tensor(num_inference_steps, C, H, W) * 2] * batch_size]

    return noise_preds
