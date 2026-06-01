from . import flux, stable_diffusion, stable_diffusion_3

__all__ = [
    "flux",
    # "stable_diffusion_xl",
    "stable_diffusion",
    "stable_diffusion_3",
    "Pipelines",
]

Pipelines = {
    # Stable Diffusion v1
    "sdv1-1": ("stable_diffusion", "CompVis/stable-diffusion-v1-1"),
    "sdv1-2": ("stable_diffusion", "CompVis/stable-diffusion-v1-2"),
    "sdv1-3": ("stable_diffusion", "CompVis/stable-diffusion-v1-3"),
    "sdv1-4": ("stable_diffusion", "CompVis/stable-diffusion-v1-4"),
    # Realistic Vision (SD v1.5)
    "realvis": ("stable_diffusion", "SG161222/Realistic_Vision_V6.0_B1_noVAE"),
    # Stable Diffusion v2
    "sdv2-1": ("stable_diffusion", "Manojb/stable-diffusion-2-1-base"),
    # Stable Diffusion v3
    "sdv3-5": ("stable_diffusion_3", "stabilityai/stable-diffusion-3.5-medium"),
    # SDXL (base)
    "sdxl-base": ("stable_diffusion_xl", "stabilityai/stable-diffusion-xl-base-1.0"),
    "sdxl-refiner": (
        "stable_diffusion_xl",
        "stabilityai/stable-diffusion-xl-refiner-1.0",
    ),
    # FLUX.1 [dev]
    "flux.1-dev": ("flux", "black-forest-labs/FLUX.1-dev"),
}
