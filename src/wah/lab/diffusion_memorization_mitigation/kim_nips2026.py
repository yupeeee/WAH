"""
How Diffusion Models Memorize
Kim et al.
NeurIPS 2026

arXiv: TBD
GitHub: TBD
"""

from typing import List, Optional, Union

import torch
import tqdm
from PIL.Image import Image

__all__ = [
    "kim_nips2026",
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

    # Compute loss (l2 norm of conditional noise prediction)
    noise_pred = pipe.predict_noise(
        latents, prompt_embeds, timesteps[0], enable_grad=True
    )
    if pipe.pipe.do_classifier_free_guidance:
        _, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text
    loss = torch.norm(noise_pred, p=2).mean()

    # Compute gradient
    (_prompt_embeds.grad,) = torch.autograd.grad(loss, [_prompt_embeds])
    _prompt_embeds.grad[:, [0]] = _prompt_embeds.grad[:, [0]] * 0
    optimizer.step()
    optimizer.zero_grad()

    _prompt_embeds = _prompt_embeds.detach()
    _prompt_embeds.requires_grad = False
    torch.cuda.empty_cache()

    return torch.cat([_negative_prompt_embeds, _prompt_embeds], dim=0)


def kim_nips2026(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
) -> List[Image]:
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
