from dataclasses import dataclass

import torch

__all__ = [
    "InitState",
    "_preprocess_prompt",
    "_get_generator",
    "_detach_clone",
]


@dataclass
class InitState:
    latents: torch.Tensor
    timesteps: torch.Tensor


def _preprocess_prompt(prompt: str | list[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("`prompt` must not be an empty list.")
        if not all(isinstance(p, str) for p in prompt):
            raise TypeError("`prompt` must be a string or a list of strings.")
        return prompt

    raise TypeError(
        f"`prompt` must be a string or a list of strings, but got {type(prompt).__name__}."
    )


def _get_generator(
    seed: int | list[int] | None,
    batch_size: int,
    device: torch.device,
) -> list[torch.Generator]:
    if seed is None:
        generator = None
    else:
        if isinstance(seed, int):
            seed = [seed] * batch_size
        generator = [torch.Generator(device=device).manual_seed(s) for s in seed]

    return generator


# RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
# ...
# To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
def _detach_clone(
    tensor: torch.Tensor,
) -> torch.Tensor:
    if not tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.clone()
    return tensor
