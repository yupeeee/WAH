"""
Classifier-Free Guidance inside the Attraction Basin May Cause Memorization
Jain et al.
CVPR 2025

arXiv: https://arxiv.org/abs/2411.16738
GitHub: https://github.com/SonyResearch/mitigating_memorization
"""

from typing import List, Optional, Union

import torch
import tqdm
from PIL.Image import Image

__all__ = [
    "jain_cvpr2025_static",
    "jain_cvpr2025_dynamic",
]


def jain_cvpr2025_static(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
    transition_t: int = 500,
    guidance_scale_before_transition: float = 0.0,
) -> List[Image]:
    # Prepare denoising
    prompt_embeds = pipe.prepare_embeds(prompt)
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=seed)

    # Denoising loop
    for i, t in tqdm.tqdm(
        enumerate(timesteps), total=num_timesteps, disable=not verbose
    ):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)

        if t < transition_t:
            noise_pred = pipe.perform_guidance(
                noise_pred, guidance_scale_before_transition
            )
        else:
            noise_pred = pipe.perform_guidance(noise_pred)

        latents = pipe.step(noise_pred, t, latents)

    # Decode image
    images = pipe.decode(latents)

    return images


def jain_cvpr2025_dynamic(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
    guidance_scale_before_transition: float = 0.0,
) -> List[Image]:
    assert (
        pipe.pipe.do_classifier_free_guidance
    ), "jain_cvpr2025_dynamic requires classifier-free guidance enabled."

    # Prepare denoising
    prompt_embeds = pipe.prepare_embeds(prompt)
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=seed)

    diff_prev = -1.0
    diff_prev_prev = -1.0

    # Denoising loop
    for i, t in tqdm.tqdm(
        enumerate(timesteps), total=num_timesteps, disable=not verbose
    ):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        diff = torch.norm(noise_pred_text - noise_pred_uncond).item()

        if diff_prev < diff_prev_prev and diff_prev > diff:
            noise_pred = pipe.perform_guidance(noise_pred)
        else:
            diff_prev_prev = diff_prev
            diff_prev = diff
            noise_pred = pipe.perform_guidance(
                noise_pred, guidance_scale_before_transition
            )

        latents = pipe.step(noise_pred, t, latents)

    # Decode image
    images = pipe.decode(latents)

    return images
