# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
import torch
import torch.nn as nn
import tqdm
from diffusers import StableDiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from PIL import ImageFilter

from ...decor import config
from ...misc.typing import (
    Any,
    Callable,
    Device,
    Dict,
    Image,
    Iterator,
    List,
    Module,
    Optional,
    Parameter,
    Tensor,
    Tuple,
    Union,
)
from ..utils import is_valid_version, load_generator, load_scheduler
from .safety_checker import SafetyChecker

__all__ = [
    "SDv1",
]

model_ids = {
    "1.1": "CompVis/stable-diffusion-v1-1",
    "1.2": "CompVis/stable-diffusion-v1-2",
    "1.3": "CompVis/stable-diffusion-v1-3",
    "1.4": "CompVis/stable-diffusion-v1-4",
    "1.5": "sd-legacy/stable-diffusion-v1-5",
}


@config
class SDv1Config:
    pipe_id: str
    prompt: Union[str, List[str]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: int = 50
    timesteps: List[int] = None
    sigmas: List[float] = None
    guidance_scale: float = 7.5
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[Tensor] = None
    prompt_embeds: Optional[Tensor] = None
    negative_prompt_embeds: Optional[Tensor] = None
    ip_adapter_image: Optional[PipelineImageInput] = None
    ip_adapter_image_embeds: Optional[List[Tensor]] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.0
    clip_skip: Optional[int] = None
    callback_on_step_end: Optional[
        Union[
            Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks
        ]
    ] = None
    callback_on_step_end_tensor_inputs: List[str] = ["latents"]
    do_classifier_free_guidance: bool = ...
    batch_size: int = ...
    lora_scale: float = ...


def _load_pipe(
    version: str,
    scheduler: str,
    **kwargs,
) -> StableDiffusionPipeline:
    assert is_valid_version(version, model_ids)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_ids[version],
        scheduler=load_scheduler(version, model_ids, scheduler),
        **kwargs,
    )
    pipe.safety_checker = None
    return pipe


def _init_config(
    pipe: StableDiffusionPipeline,
    config: SDv1Config,
) -> None:
    config.do_classifier_free_guidance = (
        config.guidance_scale > 1 and pipe.unet.config.time_cond_proj_dim is None
    )

    if isinstance(
        config.callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
    ):
        config.callback_on_step_end_tensor_inputs = (
            config.callback_on_step_end.tensor_inputs
        )

    # Default height and width to unet
    if not config.height or not config.width:
        config.height = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[0]
        )
        config.width = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[1]
        )
        config.height, config.width = (
            config.height * pipe.vae_scale_factor,
            config.width * pipe.vae_scale_factor,
        )  # to deal with lora scaling and other possible forward hooks

    # Check inputs. Raise error if not correct
    pipe.check_inputs(
        config.prompt,
        config.height,
        config.width,
        None,
        config.negative_prompt,
        config.prompt_embeds,
        config.negative_prompt_embeds,
        config.ip_adapter_image,
        config.ip_adapter_image_embeds,
        config.callback_on_step_end_tensor_inputs,
    )

    # Define call parameters
    if config.prompt is not None and isinstance(config.prompt, str):
        config.batch_size = 1
    elif config.prompt is not None and isinstance(config.prompt, list):
        config.batch_size = len(config.prompt)
    else:
        config.batch_size = config.prompt_embeds.shape[0]

    # Lora scale for cross attention
    config.lora_scale = (
        config.cross_attention_kwargs.get("scale", None)
        if config.cross_attention_kwargs is not None
        else None
    )


@torch.no_grad()
def _encode_prompt(
    pipe: StableDiffusionPipeline,
    config: SDv1Config,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
) -> Tuple[Tensor, Tensor]:
    pipe._interrupt = False
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=config.prompt if prompt is None else prompt,
        device=pipe._execution_device,
        num_images_per_prompt=config.num_images_per_prompt,
        do_classifier_free_guidance=config.do_classifier_free_guidance,
        negative_prompt=(
            config.negative_prompt if negative_prompt is None else negative_prompt
        ),
        prompt_embeds=config.prompt_embeds,
        negative_prompt_embeds=config.negative_prompt_embeds,
        lora_scale=config.lora_scale,
        clip_skip=config.clip_skip,
    )
    return prompt_embeds, negative_prompt_embeds


@torch.no_grad()
def _encode_image(
    pipe: StableDiffusionPipeline,
    config: SDv1Config,
    image: Image,
) -> Tuple[Tensor, Tensor]:
    pipe._interrupt = False
    image_embeds, uncond_image_embeds = pipe.encode_image(
        image,
        device=pipe._execution_device,
        num_images_per_prompt=config.num_images_per_prompt,
        output_hidden_states=False,
    )
    return image_embeds, uncond_image_embeds


@torch.no_grad()
def _decode_to_image(
    pipe: StableDiffusionPipeline,
    config: SDv1Config,
    latents: Union[Tensor, List[Tensor]],
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> List[Image]:
    if isinstance(latents, List):
        latents = torch.cat(latents, dim=0)
    if batch_size is None:
        batch_size = 1

    images: List[Image] = []

    for batch in tqdm.tqdm(
        latents.split(batch_size, dim=0),
        desc=f"Decode@{config.pipe_id}",
        disable=not verbose,
    ):
        decoded = pipe.vae.decode(
            batch.to(pipe._execution_device) / pipe.vae.config.scaling_factor,
            return_dict=False,
            generator=config.generator,
        )[0]
        pil_images: Image = pipe.image_processor.postprocess(
            decoded,
            output_type="pil",
            do_denormalize=[True] * decoded.shape[0],
        )
        images.extend(pil_images)

    return images


def _load_timesteps(
    pipe: StableDiffusionPipeline,
    config: SDv1Config,
) -> None:
    config.timesteps, config.num_inference_steps = retrieve_timesteps(
        scheduler=pipe.scheduler,
        num_inference_steps=config.num_inference_steps,
        device=pipe._execution_device,
        timesteps=None,
        sigmas=config.sigmas,
    )


class UNet(nn.Module):
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        config: SDv1Config,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.config = config

        self._encoder_hidden_states = ...
        self._timestep_cond = ...
        self._added_cond_kwargs = ...

    def __str__(self) -> str:
        return f"UNet@{self.config.pipe_id}"

    def to(self, device: Device) -> "UNet":
        self.pipe = self.pipe.to(device)
        return self

    def parameters(self) -> Iterator[Parameter]:
        return self.pipe.unet.parameters()

    @torch.no_grad()
    def init(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tensor:
        self.pipe._interrupt = False

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = _encode_prompt(
            self.pipe,
            self.config,
            prompt,
            negative_prompt,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.config.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if (
            self.config.ip_adapter_image is not None
            or self.config.ip_adapter_image_embeds is not None
        ):
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                self.config.ip_adapter_image,
                self.config.ip_adapter_image_embeds,
                self.pipe._execution_device,
                self.config.batch_size * self.config.num_images_per_prompt,
                self.config.do_classifier_free_guidance,
            )

        # Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            self.config.batch_size * self.config.num_images_per_prompt,
            num_channels_latents,
            self.config.height,
            self.config.width,
            prompt_embeds.dtype,
            self.pipe._execution_device,
            self.config.generator,
            None,
        )

        # Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (
                self.config.ip_adapter_image is not None
                or self.config.ip_adapter_image_embeds is not None
            )
            else None
        )

        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.config.guidance_scale - 1).repeat(
                self.config.batch_size * self.config.num_images_per_prompt
            )
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.pipe.unet.config.time_cond_proj_dim,
            ).to(device=self.pipe._execution_device, dtype=latents.dtype)

        self._encoder_hidden_states = prompt_embeds
        self._timestep_cond = timestep_cond
        self._added_cond_kwargs = added_cond_kwargs

        return latents

    def forward(self, latents: Tensor, timestep: float) -> Tensor:
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if self.config.do_classifier_free_guidance
            else latents
        )
        latent_model_input = self.pipe.scheduler.scale_model_input(
            latent_model_input, timestep
        )
        # predict the noise residual
        return self.pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=self._encoder_hidden_states,
            timestep_cond=self._timestep_cond,
            cross_attention_kwargs=self.config.cross_attention_kwargs,
            added_cond_kwargs=self._added_cond_kwargs,
            return_dict=False,
        )[0]


class SDv1(nn.Module):
    def __init__(
        self,
        version: str,
        scheduler: str,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config: SDv1Config = SDv1Config(pipe_id=f"SDv{version}_{scheduler}")

        self.pipe: StableDiffusionPipeline = _load_pipe(
            version=version,
            scheduler=scheduler,
            **kwargs,
        )
        self.device: Device = self.pipe._execution_device
        self.safety_checker = SafetyChecker(self.device)

        self.verbose: bool = verbose
        self.desc = lambda prompt: f"\033[1m{prompt}\033[0m@{self.config.pipe_id}"

        self._unet: UNet = ...
        self._seed: int = None
        self._latents: List[Tensor] = []
        self._noise_preds: List[Tensor] = []

    def __str__(self) -> str:
        config_str = []
        for k, v in self.config.__dict__.items():
            if k in ["version", "scheduler"]:
                continue
            if v is not None:
                config_str.append(f"{k}={v}")
        return f"{self.config.pipe_id}({', '.join(config_str)})"

    def init(
        self,
        **kwargs,
    ) -> None:
        r"""
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
        - `generator` (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
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
        - `ip_adapter_image`: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
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
        - `callback_on_step_end` (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        - `callback_on_step_end_tensor_inputs` (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        """
        generator = kwargs.pop(
            "generator", load_generator(seed=self._seed, device=self.device)
        )
        self.config.update(generator=generator, **kwargs)

    def to(self, device: Device) -> "SDv1":
        self.device = device
        self.pipe = self.pipe.to(self.device)
        self.safety_checker = self.safety_checker.to(self.device)
        self.config.generator = load_generator(
            seed=self._seed,
            device=self.device,
        )
        return self

    def seed(self, seed: int) -> "SDv1":
        self._seed = seed
        self.config.generator = load_generator(
            seed=self._seed,
            device=self.device,
        )
        return self

    def unet(self) -> Module:
        return UNet(self.pipe, self.config).to(self.device)

    def make_latents(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tensor:
        if prompt is not None:
            self.config.prompt = prompt
        if negative_prompt is not None:
            self.config.negative_prompt = negative_prompt
        _init_config(self.pipe, self.config)
        _load_timesteps(self.pipe, self.config)
        self._unet = self.unet()
        latents = self._unet.init(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
        )
        return latents

    @torch.no_grad()
    def forward(
        self,
        latents: Tensor,
        t: Optional[int] = 0,
    ) -> Tensor:
        self._noise_preds = []
        self._latents = [latents.cpu()]

        for _t in tqdm.tqdm(
            self.config.timesteps[: self.config.num_inference_steps - t],
            desc=self.desc(self.config.prompt),
            disable=not self.verbose,
        ):
            if self.pipe._interrupt:
                continue

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self._unet(latents, _t)
            self._noise_preds.append(noise_pred.cpu())

            # perform guidance
            if self.config.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if (
                self.config.do_classifier_free_guidance
                and self.config.guidance_rescale > 0.0
            ):
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=self.config.guidance_rescale,
                )

            # compute the previous noisy sample x_t -> x_t-1
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(
                self.config.generator, self.config.eta
            )
            latents = self.pipe.scheduler.step(
                noise_pred, _t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            # TODO: XLA is not supported yet
            # if XLA_AVAILABLE:
            #     xm.mark_step()

            self._latents.append(latents.cpu())

        # t=0: image, t=T: noise
        self._noise_preds = self._noise_preds[::-1]
        self._latents = self._latents[::-1]

        return latents

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tensor:
        return _encode_prompt(self.pipe, self.config, prompt, negative_prompt)

    @torch.no_grad()
    def encode_image(
        self,
        image: Image,
    ) -> Tensor:
        return _encode_image(self.pipe, self.config, image)

    @torch.no_grad()
    def _safety_check(
        self,
        images: List[Image],
        blur_nsfw: bool = True,
    ) -> Tuple[List[Image], List[bool]]:
        has_nsfw, _ = self.safety_checker(images)
        if blur_nsfw:
            images = [
                image.filter(ImageFilter.GaussianBlur(radius=10)) if is_nsfw else image
                for image, is_nsfw in zip(images, has_nsfw)
            ]
        return images, has_nsfw

    @torch.no_grad()
    def decode_to_image(
        self,
        latents: Tensor,
        blur_nsfw: bool = True,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[List[Image], List[bool]]:
        images = _decode_to_image(self.pipe, self.config, latents, batch_size, verbose)
        images, has_nsfw = self._safety_check(images, blur_nsfw)
        return images, has_nsfw

    @torch.no_grad()
    def pca_proj(
        self,
        latents: Union[Tensor, List[Tensor]],
        d: int = 2,
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(latents, list):
            latents = torch.stack(latents, dim=0)
        latents = latents.reshape(len(latents), -1)
        assert d <= len(latents), (
            f"d must be less or equal to len(latents) ({len(latents)}), but got {d}"
        )
        # PCA projection
        latents = latents - latents.mean(dim=0, keepdim=True)
        _, _, pca_components = torch.svd(latents, some=True)
        pca_components = pca_components[:, :d]
        latents_proj = latents @ pca_components
        return latents_proj, pca_components
