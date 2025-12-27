"""
Detecting, Explaining, and Mitigating Memorization in Diffusion Models
Wen et al.
ICLR 2024

arXiv: https://arxiv.org/abs/2407.21720
GitHub: https://github.com/YuxinWenRick/diffusion_memorization
"""

from typing import List, Optional, Union

import torch
import tqdm
from PIL.Image import Image

__all__ = [
    "wen_iclr2024",
]


def adjust_prompt(
    prompt_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    pipe,
    lr: float = 0.1,
) -> torch.Tensor:
    # Prepare prompt embeddings
    _negative_prompt_embeds, _prompt_embeds = prompt_embeds.chunk(2)
    _prompt_embeds = _prompt_embeds.clone().detach()
    _prompt_embeds.requires_grad = True
    prompt_embeds = torch.cat([_negative_prompt_embeds, _prompt_embeds], dim=0)

    # Optimizer
    optimizer = torch.optim.AdamW([_prompt_embeds], lr=lr)

    # Compute loss (l2 norm of text-conditional noise prediction)
    noise_pred = pipe.predict_noise(latents, prompt_embeds, timesteps[0], enable_grad=True)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    loss = torch.norm(noise_pred_text - noise_pred_uncond, p=2).mean()

    # Compute gradient
    (_prompt_embeds.grad,) = torch.autograd.grad(loss, [_prompt_embeds])
    _prompt_embeds.grad[:, [0]] = _prompt_embeds.grad[:, [0]] * 0
    optimizer.step()
    optimizer.zero_grad()

    _prompt_embeds = _prompt_embeds.detach()
    _prompt_embeds.requires_grad = False
    torch.cuda.empty_cache()

    return torch.cat([_negative_prompt_embeds, _prompt_embeds], dim=0)


def wen_iclr2024(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
) -> List[Image]:
    assert (
        pipe.pipe.do_classifier_free_guidance
    ), "wen_iclr2024 requires classifier-free guidance enabled."

    # Prepare denoising
    prompt_embeds = pipe.prepare_embeds(prompt)
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=seed)

    # Adjust prompt embeddings
    prompt_embeds = adjust_prompt(prompt_embeds, timesteps, latents, pipe)

    # Denoising loop
    for i, t in tqdm.tqdm(
        enumerate(timesteps), total=num_timesteps, disable=not verbose
    ):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)
        noise_pred = pipe.perform_guidance(noise_pred)
        latents = pipe.step(noise_pred, t, latents)

    # Decode image
    images = pipe.decode(latents)

    return images
