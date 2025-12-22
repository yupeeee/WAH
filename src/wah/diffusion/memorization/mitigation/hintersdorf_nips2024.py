"""
Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models
Hintersdorf et al.
NeurIPS 2024

arXiv: https://arxiv.org/abs/2406.02366
GitHub: https://github.com/ml-research/localizing_memorization_in_diffusion_models
"""

from typing import List, Optional, Union

import torch
import tqdm
from PIL.Image import Image

from ....module import getattrs, getmod

__all__ = [
    "hintersdorf_nips2024",
]


def hintersdorf_nips2024(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
) -> List[Image]:
    # Compute prompt embeddings
    prompt_embeds, empty_embeds = pipe.encode_prompt(prompt)

    # Fetch cross attention (attn2) value (to_v) layers in mid/down blocks
    value_attrs = [
        attr
        for attr in getattrs(pipe.pipe.unet)
        if "attn2.to_v" in attr and not "up_blocks" in attr
    ]
    value_layers = dict((attr, getmod(pipe.pipe.unet, attr)) for attr in value_attrs)

    # Compute activations
    with torch.no_grad():
        activations = dict(
            (attr, value_layers[attr].forward(prompt_embeds).mean(dim=1).cpu())
            for attr in value_attrs
        )  # (num_prompts, max_token_length=77, num_neurons) -(mean along dim=1)-> (num_prompts, num_neurons)
        empty_activations = dict(
            (attr, value_layers[attr].forward(empty_embeds).mean(dim=1).cpu())
            for attr in value_attrs
        )  # Instead of computing statistics for multiple prompts, we compute statistics for the empty prompt

    # Standarize activations
    standardized_activations = dict(
        (
            attr,
            (activations[attr] - empty_activations[attr].mean(dim=1, keepdim=True))
            / (empty_activations[attr].std(dim=1, keepdim=True) + 1e-6),
        )
        for attr in value_attrs
    )
    for k, v in standardized_activations.items():
        print(k, v.shape)
    exit()

    # TODO: Finish implementation!
