import torch
from diffusers import (BitsAndBytesConfig, SD3Transformer2DModel,
                       StableDiffusion3Pipeline)

from ..utils import is_valid_version

__all__ = [
    "load_pipeline",
]

model_ids = {
    "3.5-large": "stabilityai/stable-diffusion-3.5-large",
    "3.5-large-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "3.5-medium": "stabilityai/stable-diffusion-3.5-medium",
}


def load_pipeline(
    version: str,
    **kwargs,
) -> StableDiffusion3Pipeline:
    assert is_valid_version(version, model_ids)
    # nf4_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_ids[version],
        subfolder="transformer",
        # quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_ids[version],
        transformer=model_nf4,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    # pipeline.enable_model_cpu_offload()
    return pipeline
