from . import flux, stable_diffusion

__all__ = [
    "flux",
    "stable_diffusion",
    "Pipelines",
]

Pipelines = {
    # FLUX.1 [dev]
    "flux.1-dev": ("flux", "black-forest-labs/FLUX.1-dev"),
    # Stable Diffusion v1
    "sdv1-1": ("stable_diffusion", "CompVis/stable-diffusion-v1-1"),
    "sdv1-2": ("stable_diffusion", "CompVis/stable-diffusion-v1-2"),
    "sdv1-3": ("stable_diffusion", "CompVis/stable-diffusion-v1-3"),
    "sdv1-4": ("stable_diffusion", "CompVis/stable-diffusion-v1-4"),
    # Realistic Vision (SD v1.5)
    "realvis": ("stable_diffusion", "SG161222/Realistic_Vision_V6.0_B1_noVAE"),
    # Stable Diffusion v2
    "sdv2-1": ("stable_diffusion", "Manojb/stable-diffusion-2-1-base"),
}
