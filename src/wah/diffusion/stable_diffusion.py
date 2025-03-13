import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from PIL import ImageFilter

from ..misc.typing import (Device, Image, List, Literal, Optional, Tensor,
                           Tuple, Union)
from .utils import is_text

__all__ = [
    "StableDiffusion",
]


def decode_tensors(pipe, step, timestep, callback_kwargs):
    """https://huggingface.co/docs/diffusers/using-diffusers/callback"""
    latents = callback_kwargs["latents"]
    pipe.latents.append(latents.detach().cpu())
    return callback_kwargs


class StableDiffusion:
    """[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) Pipeline.

    ### Args
        - `pretrained_model_name_or_path` (str): Path to pretrained model or model identifier from huggingface.co/models.
        - `variant` (str): Variant of the model to use, e.g. "fp16" for half precision. Defaults to "fp16".
        - `blur_nsfw` (bool): Whether to blur NSFW images. Defaults to True.
        - `**kwargs`: Additional arguments to pass to the pipeline.

    ### Attributes
        - `pipe` (StableDiffusionPipeline): The underlying pipeline.
        - `blur_nsfw` (bool): Whether to blur NSFW images.
        - `safety_checker`: The safety checker model.
        - `device` (Device): The device the model is on.

    ### Example
    ```python
    >>> sd = StableDiffusion("CompVis/stable-diffusion-v1-4")
    >>> sd = sd.to("cuda")
    
    # Generate images from text prompt
    >>> images, latents, has_nsfw = sd("a photo of a cat")
    >>> images[0].save("cat.png")
    
    # Generate images from prompt embeddings
    >>> prompt_embeds, negative_prompt_embeds = sd.encode("a photo of a cat")
    >>> images, latents, has_nsfw = sd(prompt_embeds=prompt_embeds)
    >>> images[0].save("cat_from_embeds.png")
    
    # Encode images to latent space
    >>> image_latents, hidden_states = sd.encode(images, output_hidden_states=True)
    >>> print(image_latents.shape)  # [1, 4, 64, 64]
    ```
    """
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        variant: str = "fp16",
        blur_nsfw: bool = True,
        **kwargs,
    ) -> None:
        _ = kwargs.pop("safety_checker", None)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            variant=variant,
            **kwargs,
        )
        self.blur_nsfw = blur_nsfw
        self.safety_checker = self.pipe.safety_checker
        self.pipe.safety_checker = None
        self.device = self.pipe.device
        setattr(self.pipe, "generator", torch.Generator("cpu"))
        setattr(self.pipe, "latents", [])

    def to(self, device: Device):
        self.pipe.to(device)
        self.pipe.generator = torch.Generator(device=self.device)
        self.device = self.pipe.device
        return self

    @torch.no_grad()
    def _safety_check(self, images: List[Image]) -> List[bool]:
        image_to_tensor = T.PILToTensor()
        images = [image_to_tensor(image) for image in images]
        features = self.pipe.feature_extractor(images, return_tensors="pt")
        clip_input = features.pixel_values
        _, has_nsfw_concept = self.safety_checker(images=images, clip_input=clip_input)
        return has_nsfw_concept

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_embeds: Tensor = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: int = None,
        **kwargs,
    ) -> Tuple[List[Image], List[bool]]:
        assert (
            prompt is not None or prompt_embeds is not None
        ), f"Either prompt or prompt_embeds must be provided"
        if seed is not None:
            kwargs["generator"] = self.pipe.generator.manual_seed(seed)
        images = self.pipe(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            callback_on_step_end=decode_tensors,
            callback_on_step_end_tensor_inputs=["latents"],
            **kwargs,
        ).images
        has_nsfw_concept = self._safety_check(images)
        if self.blur_nsfw:
            for i, (image, is_nsfw) in enumerate(zip(images, has_nsfw_concept)):
                if is_nsfw:
                    images[i] = image.filter(ImageFilter.GaussianBlur(radius=10))
        latents = [latent for latent in torch.stack(self.pipe.latents, dim=1)]
        self.pipe.latents = []
        return images, latents, has_nsfw_concept

    @torch.no_grad()
    def encode(
        self,
        x,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt: Union[str, List[str]] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor]:
        if is_text(x):
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                x,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )
            return prompt_embeds, negative_prompt_embeds
        else:
            image_embeds, uncond_image_embeds = self.pipe.encode_image(
                x,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                output_hidden_states=output_hidden_states,
            )
            return image_embeds, uncond_image_embeds
    
    @torch.no_grad()
    def decode(
        self,
        x: Tensor,
        to: Literal["image", "text"] = "image",
    ) -> List[Union[Image, str]]:
        if to == "image":
            images = self.pipe.vae.decode(
                x.to(self.device) / self.pipe.vae.config.scaling_factor,
                return_dict=False,
                generator=self.pipe.generator,
            )[0]
            do_denormalize = [True] * images.shape[0]
            images = self.pipe.image_processor.postprocess(
                images,
                output_type="pil",
                do_denormalize=do_denormalize,
            )
            return images
        else:
            raise NotImplementedError("Text decoding is not implemented yet")
