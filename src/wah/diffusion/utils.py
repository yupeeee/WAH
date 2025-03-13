import torch
from PIL import Image

from ..misc.typing import Tensor

__all__ = [
    "is_text",
    "sdxl2rgb",
]


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
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)
