"""
Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models
Hintersdorf et al.
NeurIPS 2024

arXiv: https://arxiv.org/abs/2406.02366
GitHub: https://github.com/ml-research/localizing_memorization_in_diffusion_models
"""

import copy

# import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import tqdm
from PIL.Image import Image
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from ....module import getattrs, getmod

# from ....web.download import from_url, md5_check
from .hintersdorf_nips2024_hook import BlockActivations
from .hintersdorf_nips2024_stats import sdv1_4_means, sdv1_4_stds

__all__ = [
    "hintersdorf_nips2024",
]


# SDV1_4_STATS_URL = "https://raw.githubusercontent.com/ml-research/localizing_memorization_in_diffusion_models/main/statistics/statistics_additional_laion_prompts_v1_4.pt"
# SDV1_4_STATS_CHECKSUM = "cfe00e5b554c6f516cfcfe4468896dcb"


# def load_sdv1_4_stats() -> Tuple[torch.Tensor, torch.Tensor]:
#     sdv1_4_stats_path = from_url(SDV1_4_STATS_URL, root="hintersdorf_nips2024_stats")
#     assert md5_check(sdv1_4_stats_path, SDV1_4_STATS_CHECKSUM)
#     sdv1_4_means, sdv1_4_stds = torch.load(
#         sdv1_4_stats_path, weights_only=True, map_location="cpu"
#     )
#     # os.remove(sdv1_4_stats_path)
#     return sdv1_4_means, sdv1_4_stds


# sdv1_4_means, sdv1_4_stds = load_sdv1_4_stats()


def compute_noise_differences(
    prompt_embeds: torch.Tensor,
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    blocking_indices: Dict[str, List[int]] = None,
):
    # Register hooks to block specified neurons
    if blocking_indices is not None:
        hooks = []

        for attr, indices in blocking_indices.items():
            hook = getmod(pipe.pipe.unet, attr).register_forward_hook(
                BlockActivations(indices)
            )
            hooks.append(hook)

    # Predict noise at t=T
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=seed)
    noise_preds = pipe.predict_noise(latents, prompt_embeds, timesteps[0])
    noise_preds = pipe.perform_guidance(noise_preds)

    # Remove hooks
    if blocking_indices is not None:
        for hook in hooks:
            hook.remove()

    # Compute noise_pred - latents at t=T
    noise_diffs = noise_preds - latents
    min_values = noise_diffs.amin(dim=[2, 3])
    max_values = noise_diffs.amax(dim=[2, 3])
    noise_diffs = (noise_diffs - min_values.unsqueeze(-1).unsqueeze(-1)) / (
        max_values - min_values
    ).unsqueeze(-1).unsqueeze(-1)

    return noise_diffs


def get_ood_neurons(
    activations: Dict[str, torch.Tensor],
    theta: float,
    k: int,
) -> Dict[str, List[int]]:
    blocking_indices = dict((attr, []) for attr in activations.keys())

    for attr, activation in activations.items():
        # Check each neuron in each layer for OOD activation
        indices = (activation.abs() > theta).nonzero(as_tuple=True)[0].tolist()

        # Add k neurons of layer l with the highest absolute activations to the candidate set
        topk_indices = activation.abs().topk(k=min(k, len(activation))).indices
        if topk_indices.numel() > 0:
            indices += [e.item() for e in topk_indices]

        blocking_indices[attr] = indices

    return blocking_indices


def compute_memorization(
    noise_diffs_unblocked: torch.Tensor,
    noise_diffs_blocked: torch.Tensor,
) -> torch.Tensor:
    ssim = MultiScaleStructuralSimilarityIndexMeasure(
        reduction="none",
        kernel_size=11,
        betas=(0.33, 0.33, 0.33),
    )(noise_diffs_unblocked, noise_diffs_blocked)

    return ssim.item()


def initial_neuron_selection(
    activations: Dict[str, torch.Tensor],
    prompt_embeds: torch.Tensor,
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    ssim_threshold: float = 0.428,
    theta_min: float = 1.0,
) -> Tuple[Dict[str, List[int]], float]:
    ssim = 1.0  # Initialize memorization score as maximum
    theta = 5.0  # Initialize threshold of OOD activation detection
    k = 0  # Initialize k for top-k activation detection
    ssim_threshold_ref = (
        ssim_threshold  # Set refinement memorization threshold to current threshold
    )

    # Compute noise differences with all neurons activated
    noise_diffs_unblocked = compute_noise_differences(
        prompt_embeds, pipe, seed, blocking_indices=None
    )

    # Increase set of candidate neurons until target memorization score is reached
    while ssim > ssim_threshold:  # While memorization score above threshold
        blocking_indices = get_ood_neurons(
            activations, theta, k
        )  # Detect neurons with OOD activations
        noise_diffs_blocked = compute_noise_differences(
            prompt_embeds, pipe, seed, blocking_indices
        )  # Compute noise differences
        ssim = compute_memorization(
            noise_diffs_unblocked, noise_diffs_blocked
        )  # Compute memorization score (SSIM)

        if theta < theta_min:  # Minimum activation threshold not reached
            ssim_threshold_ref = (
                ssim  # Set refinement threshold to current memorization score
            )
            break  # Stop if activation threshold is too low

        # Adjust OOD detection parameters to increase set of candidate neurons
        theta = theta - 0.25  # Decrease threshold for OOD detection
        k += 1

    return blocking_indices, ssim_threshold_ref


def neuron_selection_refinement(
    activations: Dict[str, torch.Tensor],
    blocking_indices: Dict[str, List[int]],
    ssim_threshold_ref: float,
    prompt_embeds: torch.Tensor,
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
) -> Dict[str, List[int]]:
    # Compute noise differences with all neurons activated
    noise_diffs_unblocked = compute_noise_differences(
        prompt_embeds, pipe, seed, blocking_indices=None
    )

    # 1) Remove all layers with no blocked neurons or neurons without any impact
    # Check all candidate neurons of individual layers at once for memorization
    for layer in reversed(
        list(blocking_indices.keys())
    ):  # Iterate over all layers to remove low impact layers
        if len(blocking_indices[layer]) == 0:
            del blocking_indices[layer]
        else:
            current_blocking_indices = copy.deepcopy(blocking_indices)
            current_blocking_indices[layer] = (
                []
            )  # Compute set of neurons without neurons of layer l
            noise_diffs_blocked = compute_noise_differences(
                prompt_embeds, pipe, seed, current_blocking_indices
            )
            ssim = compute_memorization(noise_diffs_unblocked, noise_diffs_blocked)
            if ssim < ssim_threshold_ref:  # Minimum memorization threshold not reached
                del blocking_indices[layer]  # Remove neurons of layer l from neuron set

    # 2) Remove all neurons with no impact
    # Check all remaining candidate neurons individually
    for layer in reversed(
        list(blocking_indices.keys())
    ):  # Iterate over each remaining layer
        for neuron in blocking_indices[layer]:
            current_blocking_indices = copy.deepcopy(blocking_indices)
            current_blocking_indices[layer].remove(
                neuron
            )  # Compute set of neurons without neuron n
            noise_diffs_blocked = compute_noise_differences(
                prompt_embeds, pipe, seed, current_blocking_indices
            )
            ssim = compute_memorization(noise_diffs_unblocked, noise_diffs_blocked)
            if ssim < ssim_threshold_ref:  # Minimum memorization threshold not reached
                blocking_indices[layer].remove(neuron)  # Remove current neuron from set

    return blocking_indices


def hintersdorf_nips2024_single_prompt(
    prompt: str,
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
) -> List[Image]:
    if isinstance(prompt, list):
        assert (
            len(prompt) == 1
        ), f"Mitigation for multiple prompts is not supported for hintersdorf_nips2024."
        prompt = prompt[0]
    assert isinstance(prompt, str), f"prompt must be a string, got type {type(prompt)}"

    # Fetch cross attention (attn2) value (to_v) layers in down & mid blocks
    value_attrs = [
        attr
        for attr in getattrs(pipe.pipe.unet)
        if "attn2.to_v" in attr and not "up_blocks" in attr
    ]
    value_layers = dict((attr, getmod(pipe.pipe.unet, attr)) for attr in value_attrs)

    means = dict((attr, sdv1_4_means[i]) for i, attr in enumerate(value_attrs))
    stds = dict((attr, sdv1_4_stds[i]) for i, attr in enumerate(value_attrs))

    # Compute activations
    prompt_embeds, _ = pipe.encode_prompt(prompt)
    with torch.no_grad():
        activations = dict(
            (attr, value_layers[attr].forward(prompt_embeds).mean(dim=1)[0].cpu())
            for attr in value_attrs
        )  # (num_prompts, max_token_length=77, num_neurons) -(mean along dim=1)-> (num_prompts, num_neurons) with num_prompts = 1

    # Standarize activations
    for attr in value_attrs:
        activations[attr] = (activations[attr] - means[attr]) / stds[attr]

    # Initial neuron selection
    prompt_embeds = pipe.prepare_embeds(prompt)
    blocking_indices, ssim_threshold_ref = initial_neuron_selection(
        activations, prompt_embeds, pipe, seed
    )

    # Neuron selection refinement
    blocking_indices = neuron_selection_refinement(
        activations, blocking_indices, ssim_threshold_ref, prompt_embeds, pipe, seed
    )

    # Prepare denoising
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=seed)

    # Register hooks to block specified neurons
    hooks = []
    for attr, indices in blocking_indices.items():
        hook = getmod(pipe.pipe.unet, attr).register_forward_hook(
            BlockActivations(indices)
        )
        hooks.append(hook)

    # Denoising loop
    for i, t in tqdm.tqdm(
        enumerate(timesteps), total=num_timesteps, disable=not verbose
    ):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)
        noise_pred = pipe.perform_guidance(noise_pred)
        latents = pipe.step(noise_pred, t, latents)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Decode image
    images = pipe.decode(latents)

    return images


def hintersdorf_nips2024(
    prompt: List[str],
    pipe,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = False,
) -> List[Image]:
    if isinstance(prompt, str):
        prompt = [prompt]
    images = []
    for p in prompt:
        images.extend(hintersdorf_nips2024_single_prompt(p, pipe, seed, verbose))
    return images
