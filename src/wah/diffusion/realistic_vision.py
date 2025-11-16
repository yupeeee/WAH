from .stable_diffusion import StableDiffusion

__all__ = [
    "RealisticVision",
]


SUPPORTED_VERSIONS = {
    "1.4": "SG161222/Realistic_Vision_V1.4",
}


class RealisticVision(StableDiffusion):
    """
    [Realistic Vision](https://civitai.com/) Pipeline for text-to-image generation.

    ## Example (Pipeline call)
    ```python
    import wah

    # Load pipeline
    pipe = wah.diffusion.RealisticVision(
        version="1.4",
        scheduler="DDIM",
    )
    pipe.to("cuda")
    pipe.verbose(True)

    # Denoising
    prompt = "a photo of an astronaut riding a horse on mars"
    images = pipe(prompt).images
    images[0].save("image.png")
    ```

    ## Example (Manual denoising)
    ```python
    import wah

    # Load pipeline
    pipe = wah.diffusion.RealisticVision(
        version="1.4",
        scheduler="DDIM",
    )

    # Initialize
    pipe.to("cuda")
    pipe.init(
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # Prepare denoising
    prompt = "a photo of an astronaut riding a horse on mars"
    prompt_embeds = pipe.prepare_embeds(prompt)
    timesteps, num_timesteps, num_warmup_steps = pipe.prepare_timesteps()
    latents = pipe.prepare_latents(prompt_embeds, seed=0)

    # Denoising loop
    for i, t in enumerate(timesteps):
        noise_pred = pipe.predict_noise(latents, prompt_embeds, t)
        noise_pred = pipe.perform_guidance(noise_pred)
        latents = pipe.step(noise_pred, t, latents)

    # Decode image
    images = pipe.decode(latents)
    images[0].save("image.png")
    ```
    """

    _SUPPORTED_VERSIONS = SUPPORTED_VERSIONS

    def __init__(
        self,
        version: str,
        scheduler: str,
        variant: str = None,
        verbose: bool = False,
        safety_check: bool = True,
        **kwargs,
    ):
        super().__init__(
            version,
            scheduler,
            variant=variant,
            verbose=verbose,
            safety_check=safety_check,
            **kwargs,
        )
