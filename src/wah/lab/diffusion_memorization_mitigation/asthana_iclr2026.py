"""
Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability
Asthana et al.
ICLR 2026

arXiv: https://arxiv.org/abs/2601.20642
GitHub: https://github.com/rohanasthana/memorization-anisotropy
"""

from typing import List, Optional, Union

import torch
import tqdm
from PIL.Image import Image

__all__ = [
    "asthana_iclr2026",
]


def _mitigate_latents(
    prompt_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    pipe,
    *,
    miti_thres: float = 8.2,
    miti_budget: int = 8,
    miti_lr: float = 0.05,
    miti_max_steps: int = 10,
    gen_seed: int = 0,
) -> torch.Tensor:
    if not pipe.pipe.do_classifier_free_guidance:
        raise ValueError("asthana_iclr2026_mitigate requires classifier-free guidance.")

    if pipe.version.split(".")[0] == "1":
        sd_ver = 1
    else:
        sd_ver = 0

    ipp = 1  # images per prompt
    thres = miti_thres
    seed = gen_seed

    p_uncond, p_cond = prompt_embeds[0].unsqueeze(dim=0), prompt_embeds[ipp].unsqueeze(
        dim=0
    )
    p_tot = torch.cat(
        [p_uncond.detach()] * miti_budget + [p_cond.detach()] * miti_budget
    )

    t = timesteps[0]
    beta_prod = torch.sqrt(1 - pipe.pipe.scheduler.alphas_cumprod[t])
    beta = torch.sqrt(pipe.pipe.scheduler.alphas_cumprod[t])

    lat_lst = []
    counter = 0  # counter for updating threshold
    while True:
        torch.manual_seed(seed)
        lat = torch.randn(
            (miti_budget, *latents.shape[1:]), device=latents.device, requires_grad=True
        )
        lat_out = pipe.pipe.scheduler.scale_model_input(lat, t)
        lat_single = torch.cat([lat_out] * 2)
        optimizer = torch.optim.Adam([lat], lr=miti_lr)

        step_cnt = 0
        indice_record = torch.tensor([], device=lat.device, dtype=torch.long)
        while step_cnt < miti_max_steps:

            noise = pipe.pipe.unet(lat_single, t, encoder_hidden_states=p_tot)[0]
            uc_pred, c_pred = (
                noise.chunk(2)
                if sd_ver == 1
                else (-(beta * noise - lat_single) / beta_prod).chunk(2)
            )
            diff_pred = c_pred - uc_pred
            diff_pred = (
                diff_pred
                / diff_pred.view(miti_budget, -1).norm(dim=1)[:, None, None, None]
            )

            lat_modi = torch.cat([lat + diff_pred] * 2)
            noise_modi = pipe.pipe.unet(lat_modi, t, encoder_hidden_states=p_tot)

            # calculate proxy for SAIL loss (refer to Algorithm2 of https://arxiv.org/pdf/2412.04140)
            if sd_ver == 1:
                uc_modi, c_modi = noise_modi[0].chunk(2)
                hvp_loss = (c_modi - uc_modi).view(miti_budget, -1).norm(dim=1)
                gaussianity = lat.view(miti_budget, -1).norm(dim=1)
                loss = hvp_loss + 0.05 * gaussianity
            else:
                uc_modi, c_modi = (
                    -(beta * noise_modi[0] - lat_modi) / beta_prod
                ).chunk(2)
                hvp_loss = (c_modi - uc_modi).view(miti_budget, -1).norm(dim=1)
                gaussianity = lat.view(miti_budget, -1).norm(dim=1)
                loss = hvp_loss + 0.01 * gaussianity

            indices = torch.where(loss <= thres)[0]
            updated_indices = torch.cat([indice_record, indices]).unique()

            if len(updated_indices) > len(indice_record):
                new_indices = updated_indices[
                    ~torch.isin(updated_indices, indice_record)
                ]
                lat_lst.extend([lat[i].detach().unsqueeze(0) for i in new_indices])
                indice_record = updated_indices
                # print(f"{len(new_indices)} Latents Added")

                if len(lat_lst) >= ipp:
                    break

            loss = loss.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_cnt += 1

        if len(lat_lst) >= ipp:
            break
        else:
            counter += 1

        # This code is to increase l_thres & change 'seed' to prevent model
        # when model is not able to find satisfiable latents
        if counter // 3 != (counter - 1) // 3:
            thres += 0.1
            seed = torch.randint(0, 50000, (1,)).item()

            # print(f'Thres Updated: {thres:.2f}')
            # print(f'Seed Changed: {seed}')

    torch.cuda.empty_cache()
    latents = torch.cat(lat_lst)[:ipp]

    return latents


def asthana_iclr2026_mitigate(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
    *,
    miti_thres: float = 8.2,
    miti_budget: int = 8,
    miti_lr: float = 0.05,
    miti_max_steps: int = 10,
) -> List[Image]:
    if isinstance(prompt, str):
        prompt = [prompt]
    prompt_embeds = pipe.prepare_embeds(prompt)
    prompt_embeds_uncond, prompt_embeds_cond = prompt_embeds.chunk(2, dim=0)
    timesteps, num_timesteps, _ = pipe.prepare_timesteps()

    latents = []
    for i in range(len(prompt)):
        prompt_embed = torch.stack(
            [prompt_embeds_uncond[i], prompt_embeds_cond[i]], dim=0
        )

        latent = pipe.prepare_latents(prompt_embed, seed=seed)
        latent = _mitigate_latents(
            prompt_embed,
            timesteps,
            latent,
            pipe,
            miti_thres=miti_thres,
            miti_budget=miti_budget,
            miti_lr=miti_lr,
            miti_max_steps=miti_max_steps,
            gen_seed=seed,
        )
        latents.append(latent)
    latents = torch.cat(latents, dim=0)

    for _, t in tqdm.tqdm(
        enumerate(timesteps), total=num_timesteps, disable=not verbose
    ):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)
        noise_pred = pipe.perform_guidance(noise_pred)
        latents = pipe.step(noise_pred, t, latents)

    return pipe.decode(latents)


def asthana_iclr2026(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
) -> List[Image]:
    return asthana_iclr2026_mitigate(prompt, pipe, seed, verbose)
