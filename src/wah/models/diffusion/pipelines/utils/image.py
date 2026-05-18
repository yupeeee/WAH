from torch import Tensor

__all__ = [
    "_get_image_embeds",
]


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L514
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L390
def _get_image_embeds(
    pipe,
    image,
    num_images_per_prompt: int = 1,
    # output_hidden_states: bool = False,
) -> Tensor:
    device = pipe._execution_device
    dtype = next(pipe.image_encoder.parameters()).dtype

    if not isinstance(image, Tensor):
        image = pipe.feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    image_embeds = pipe.image_encoder(image).image_embeds
    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    return image_embeds
