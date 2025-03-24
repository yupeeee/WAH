import torch
from diffusers import StableDiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.utils import deprecate

from ...misc.typing import Tensor, Tuple
from ..utils import is_valid_version
from .scheduler import load_scheduler

__all__ = [
    "load_pipeline",
    "noise_at_T",
]

model_ids = {
    "1.1": "CompVis/stable-diffusion-v1-1",
    "1.2": "CompVis/stable-diffusion-v1-2",
    "1.3": "CompVis/stable-diffusion-v1-3",
    "1.4": "CompVis/stable-diffusion-v1-4",
    "1.5": "sd-legacy/stable-diffusion-v1-5",
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


def noise_at_T(
    pipe: StableDiffusionPipeline,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py"""
    prompt = kwargs.pop("prompt", None)
    height = kwargs.pop("height", None)
    width = kwargs.pop("width", None)
    num_inference_steps = kwargs.pop("num_inference_steps", 50)
    timesteps = kwargs.pop("timesteps", None)
    sigmas = kwargs.pop("sigmas", None)
    guidance_scale = kwargs.pop("guidance_scale", 7.5)
    negative_prompt = kwargs.pop("negative_prompt", None)
    num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
    eta = kwargs.pop("eta", 0.0)
    generator = kwargs.pop("generator", pipe.generator)
    latents = kwargs.pop("latents", None)
    prompt_embeds = kwargs.pop("prompt_embeds", None)
    negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
    ip_adapter_image = kwargs.pop("ip_adapter_image", None)
    ip_adapter_image_embeds = kwargs.pop("ip_adapter_image_embeds", None)
    output_type = kwargs.pop("output_type", "pil")
    return_dict = kwargs.pop("return_dict", True)
    cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", None)
    guidance_rescale = kwargs.pop("guidance_rescale", 0.0)
    clip_skip = kwargs.pop("clip_skip", None)
    callback_on_step_end = kwargs.pop("callback_on_step_end", None)
    callback_on_step_end_tensor_inputs = kwargs.pop(
        "callback_on_step_end_tensor_inputs", ["latents"]
    )

    do_classifier_free_guidance = (
        guidance_scale > 1 and pipe.unet.config.time_cond_proj_dim is None
    )

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

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
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )
    pipe._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    # 3. Encode input prompt
    lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # expand the latents if we are doing classifier free guidance
    t = timesteps[0]
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # UNet args
    unet_args = {
        "sample": latent_model_input,
        "timestep": t,
        "encoder_hidden_states": prompt_embeds,
        "timestep_cond": timestep_cond,
        "cross_attention_kwargs": cross_attention_kwargs,
        "added_cond_kwargs": added_cond_kwargs,
        "return_dict": False,
    }

    # predict the noise residual
    noise_pred = pipe.unet(**unet_args)[0]

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    if do_classifier_free_guidance and guidance_rescale > 0.0:
        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        noise_pred = rescale_noise_cfg(
            noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
        )

    return latent_model_input, noise_pred
