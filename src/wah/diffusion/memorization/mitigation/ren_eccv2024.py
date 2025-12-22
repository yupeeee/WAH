"""
Unveiling and Mitigating Memorization in Text-to-image Diffusion Models through Cross Attention
Ren et al.
ECCV 2024

arXiv: https://arxiv.org/abs/2403.11052
GitHub: https://github.com/renjie3/MemAttn
"""

from typing import List, Optional, Union

import tqdm
from PIL.Image import Image

from ....module import getattrs, getmod
from .ren_eccv2024_attn import AttnProcessor2_0

__all__ = [
    "ren_eccv2024",
]


def ren_eccv2024(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
    c1: float = 1.25,
) -> List[Image]:
    # Compute prompt length
    # Tokenize prompt(s) just to get lengths
    text_inputs = pipe.pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Count non-padding tokens in the input_ids
    pad_token_id = pipe.pipe.tokenizer.pad_token_id
    # Each row is a prompt; sum for each (77 - num_pad_tokens)
    prompt_lengths = (text_inputs.input_ids != pad_token_id).sum(dim=1)
    prompt_lengths = (prompt_lengths + 1).tolist()  # +1 for the beginning token

    # Prepare denoising
    prompt_embeds = pipe.prepare_embeds(prompt)
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=seed)

    # Replace attn processor of cross attention layers
    # to rescale beginning token logits and mask out summary tokens
    ca_attrs = [
        attr.split(".attn2")[0] + ".attn2"
        for attr in getattrs(pipe.pipe.unet)
        if "attn2" in attr
    ]
    for ca_attr in ca_attrs:
        getmod(pipe.pipe.unet, ca_attr).set_processor(
            AttnProcessor2_0(
                prompt_lengths=prompt_lengths,
                max_length=pipe.pipe.tokenizer.model_max_length,
                c1=c1,
            )
        )

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
