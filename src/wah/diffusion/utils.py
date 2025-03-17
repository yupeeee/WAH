import subprocess

import torch
from PIL import Image

from ..misc.typing import Dict, Tensor

__all__ = [
    "login",
    "is_valid_version",
    "is_text",
    "sdxl2rgb",
]


def login() -> None:
    """Login to Hugging Face using access token via CLI.

    ### Example
    ```python
    >>> login()
    ```
    """
    # Check if already logged in by attempting to run whoami
    result = subprocess.run(
        ["huggingface-cli", "whoami"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout == "Not logged in\n":
        # If not logged in, login
        process = subprocess.Popen(
            ["huggingface-cli", "login"],
            stdin=subprocess.PIPE,
            text=True,
        )
        process.communicate(input="n\n")
        process.wait()
    else:
        print("Logged in as", result.stdout.strip())


def is_valid_version(version: str, ids: Dict[str, str]) -> bool:
    """Check if a version is valid for a given model ID."""
    if version in ids.keys():
        return True
    else:
        raise ValueError(
            f"Invalid version ({version})\n"
            f"Valid versions and their model IDs are:\n" +
            "\n".join(f"{k} ({v})" for k, v in ids.items())
        )


def is_text(x):
    """Returns True if x is a string or list of strings (Union[str, List[str]]), False otherwise"""
    return isinstance(x, (str, list)) and (
        isinstance(x, str) or all(isinstance(s, str) for s in x)
    )


def sdxl2rgb(latents: Tensor) -> Tensor:
    """https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space"""
    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35),
    )

    weights_tensor = torch.t(
        torch.tensor(weights, dtype=latents.dtype).to(latents.device)
    )
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(
        latents.device
    )
    rgb_tensor = torch.einsum(
        "...lxy,lr -> ...rxy", latents, weights_tensor
    ) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(
        0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)
