import torch
import tqdm
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput

from ...misc.typing import (
    Any,
    Device,
    Dict,
    Image,
    List,
    Optional,
    Tensor,
    Tuple,
    Union,
)
from . import v1, v2
from .safety_checker import SafetyChecker
from .utils import load_generator

__all__ = [
    "StableDiffusion",
]

supported_versions = [
    version
    for version in list(v1.model_ids.keys()) + list(v2.model_ids.keys())
    # + list(v3_5.model_ids.keys())
]


def load_pipe(
    version: str,
    scheduler: str,
    **kwargs,
) -> Union[StableDiffusionPipeline, StableDiffusion3Pipeline]:
    if version not in supported_versions:
        raise ValueError(
            f"Version {version} is not supported for Stable Diffusion.\n"
            f"Supported versions: {supported_versions}"
        )
    if version.startswith("1."):
        return v1._load_pipe(version, scheduler, **kwargs)
    elif version.startswith("2"):
        return v2._load_pipe(version, scheduler, **kwargs)
    # elif version.startswith("3.5-"):
    #     return sd3_5.load_pipeline(version, scheduler, verbose, **kwargs)
    else:
        raise


class StableDiffusion:
    def __init__(
        self,
        version: str,
        scheduler: str,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        self._version = version
        self._scheduler = scheduler
        self._seed: int = None

        self.pipe = load_pipe(version, scheduler, **kwargs)
        if not verbose:
            self.pipe.set_progress_bar_config(disable=True)
        self.device: Device = self.pipe._execution_device
        self._generator: torch.Generator = load_generator(self._seed, self.device)
        self.safety_checker = SafetyChecker(self.device)

        self.latents: List[Tensor] = []
        self.noise_preds: List[Tensor] = []

    def __str__(self) -> str:
        return f"SDv{self._version}_{self._scheduler}"

    def __repr__(self) -> str:
        return self.__str__()

    def seed(self, _seed: int) -> "StableDiffusion":
        self._seed = _seed
        self._generator = load_generator(self._seed, self.device)
        return self

    def to(self, device: Device) -> "StableDiffusion":
        self.device = device
        self.pipe.to(device)
        self._generator = load_generator(self._seed, self.device)
        self.safety_checker = self.safety_checker.to(device)
        return self

    def verbose(self, _verbose: bool) -> "StableDiffusion":
        if not _verbose:
            self.pipe.set_progress_bar_config(disable=True)
        return self

    def _latents_callback(self, pipe, step, timestep, callback_kwargs):
        self.latents.append(callback_kwargs["latents"])
        return callback_kwargs

    def _unet_hook(self, module, input, output):
        self.noise_preds.append(output[0] if isinstance(output, tuple) else output)

    @torch.no_grad()
    def _decode(
        self,
        latents: Tensor,
    ) -> List[Image]:
        decoded = self.pipe.vae.decode(
            latents.to(self.pipe._execution_device)
            / self.pipe.vae.config.scaling_factor,
            return_dict=False,
            generator=self._generator,
        )[0]
        pil_images = self.pipe.image_processor.postprocess(
            decoded.detach(),
            output_type="pil",
            do_denormalize=[True] * decoded.shape[0],
        )
        return pil_images

    def decode(
        self,
        latents: Tensor,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Image]:
        latents_batches = torch.split(
            latents, batch_size if batch_size is not None else 1
        )
        images = []
        for latents_batch in tqdm.tqdm(
            latents_batches,
            desc="Decoding",
            disable=not verbose,
        ):
            images.extend(self._decode(latents_batch))
        return images

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[Image], List[Tensor], List[Tensor]]:
        r"""
        The call function to the pipeline for generation.

        ### Args
        - `prompt` (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        - `height` (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        - `width` (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        - `num_inference_steps` (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        - `timesteps` (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        - `sigmas` (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        - `guidance_scale` (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        - `negative_prompt` (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        - `num_images_per_prompt` (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        - `eta` (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        - `latents` (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        - `prompt_embeds` (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        - `negative_prompt_embeds` (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        - `ip_adapter_image` (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        - `ip_adapter_image_embeds` (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
            contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        - `output_type` (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        - `return_dict` (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        - `cross_attention_kwargs` (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        - `guidance_rescale` (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
            using zero terminal SNR.
        - `clip_skip` (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        hook = self.pipe.unet.register_forward_hook(self._unet_hook)
        with torch.no_grad():
            images = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=self._generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                callback_on_step_end=self._latents_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                **kwargs,
            ).images
        hook.remove()

        # Rearrange from [(B,*D)]*T to [(T,*D)]*B
        latents: List[Tensor] = [
            torch.stack([latents[b] for latents in self.latents]).cpu()
            for b in range(self.latents[0].shape[0])
        ]
        noise_preds: List[Tensor] = [
            torch.stack([noise_preds[b] for noise_preds in self.noise_preds]).cpu()
            for b in range(self.noise_preds[0].shape[0])
        ]
        self.latents = []
        self.noise_preds = []
        return (
            images,
            latents,
            (
                noise_preds[: len(latents)],  # noise_preds_uncond
                noise_preds[len(latents) :],  # noise_preds_text
            ),
        )
