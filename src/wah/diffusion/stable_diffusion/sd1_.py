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
from diffusers.utils import is_torch_xla_available

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
from ..utils import is_valid_version, load_generator
from .safety_checker import SafetyChecker
from .scheduler import load_scheduler

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

# TODO: XLA is not supported yet
# if is_torch_xla_available():
#     import torch_xla.core.xla_model as xm

#     XLA_AVAILABLE = True
# else:
#     XLA_AVAILABLE = False


def load_pipeline(
    version: str,
    scheduler: str,
    **kwargs,
) -> StableDiffusionPipeline:
    assert is_valid_version(version, model_ids)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_ids[version],
        scheduler=load_scheduler(version, model_ids, scheduler),
        **kwargs,
    )
    pipeline.safety_checker = None
    return pipeline


@config
class SDv1Config:
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


def load_params(
    pipe: StableDiffusionPipeline,
    config: SDv1Config,
) -> Dict[str, Any]:
    prompt = config.prompt
    height = config.height
    width = config.width
    num_inference_steps = config.num_inference_steps
    timesteps = config.timesteps
    sigmas = config.sigmas
    guidance_scale = config.guidance_scale
    negative_prompt = config.negative_prompt
    num_images_per_prompt = config.num_images_per_prompt
    eta = config.eta
    generator = config.generator
    latents = config.latents
    prompt_embeds = config.prompt_embeds
    negative_prompt_embeds = config.negative_prompt_embeds
    ip_adapter_image = config.ip_adapter_image
    ip_adapter_image_embeds = config.ip_adapter_image_embeds
    output_type = config.output_type
    return_dict = config.return_dict
    cross_attention_kwargs = config.cross_attention_kwargs
    guidance_rescale = config.guidance_rescale
    clip_skip = config.clip_skip
    callback_on_step_end = config.callback_on_step_end
    callback_on_step_end_tensor_inputs = config.callback_on_step_end_tensor_inputs

    do_classifier_free_guidance = (
        guidance_scale > 1 and pipe.unet.config.time_cond_proj_dim is None
    )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # Default height and width to unet
    if not height or not width:
        height = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[0]
        )
        width = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[1]
        )
        height, width = height * pipe.vae_scale_factor, width * pipe.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # Check inputs. Raise error if not correct
    pipe.check_inputs(
        prompt,
        height,
        width,
        None,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Lora scale for cross attention
    lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )

    return {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "timesteps": timesteps,
        "sigmas": sigmas,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": num_images_per_prompt,
        "eta": eta,
        "generator": generator,
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "ip_adapter_image": ip_adapter_image,
        "ip_adapter_image_embeds": ip_adapter_image_embeds,
        "output_type": output_type,
        "return_dict": return_dict,
        "cross_attention_kwargs": cross_attention_kwargs,
        "guidance_rescale": guidance_rescale,
        "clip_skip": clip_skip,
        "callback_on_step_end": callback_on_step_end,
        "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
        "do_classifier_free_guidance": do_classifier_free_guidance,
        "batch_size": batch_size,
        "lora_scale": lora_scale,
    }


@torch.no_grad()
def encode_prompt(
    pipe: StableDiffusionPipeline,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    prompt = kwargs.get("prompt", None)
    negative_prompt = kwargs.get("negative_prompt", None)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    prompt_embeds = kwargs.get("prompt_embeds", None)
    negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
    clip_skip = kwargs.get("clip_skip", None)
    do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", False)
    lora_scale = kwargs.get("lora_scale", None)

    pipe._interrupt = False
    device = pipe._execution_device

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=clip_skip,
    )
    return prompt_embeds, negative_prompt_embeds


@torch.no_grad()
def encode_image(
    pipe: StableDiffusionPipeline,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    image = kwargs.get("image", None)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    output_hidden_states = kwargs.get("output_hidden_states", False)

    pipe._interrupt = False
    device = pipe._execution_device

    image_embeds, uncond_image_embeds = pipe.encode_image(
        image,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        output_hidden_states=output_hidden_states,
    )
    return image_embeds, uncond_image_embeds


@torch.no_grad()
def decode_to_image(
    pipe: StableDiffusionPipeline,
    latents: Tensor,
    generator: torch.Generator,
) -> List[Image]:
    images = pipe.vae.decode(
        latents.to(pipe._execution_device) / pipe.vae.config.scaling_factor,
        return_dict=False,
        generator=generator,
    )[0]
    do_denormalize = [True] * images.shape[0]
    images = pipe.image_processor.postprocess(
        images,
        output_type="pil",
        do_denormalize=do_denormalize,
    )
    return images


def load_timesteps(
    pipe: StableDiffusionPipeline,
    **kwargs,
) -> Tuple[Tensor, int]:
    num_inference_steps = kwargs["num_inference_steps"]
    timesteps = kwargs["timesteps"]
    sigmas = kwargs["sigmas"]
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler=pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=pipe._execution_device,
        timesteps=timesteps,
        sigmas=sigmas,
    )
    return timesteps, num_inference_steps


@torch.no_grad()
def prepare_unet_inputs(
    pipe: StableDiffusionPipeline,
    **kwargs,
) -> Tuple[Tensor, Dict[str, Any]]:
    height = kwargs["height"]
    width = kwargs["width"]
    guidance_scale = kwargs["guidance_scale"]
    num_images_per_prompt = kwargs["num_images_per_prompt"]
    generator = kwargs["generator"]
    latents = kwargs["latents"]
    prompt_embeds = kwargs["prompt_embeds"]
    negative_prompt_embeds = kwargs["negative_prompt_embeds"]
    ip_adapter_image = kwargs["ip_adapter_image"]
    ip_adapter_image_embeds = kwargs["ip_adapter_image_embeds"]
    cross_attention_kwargs = kwargs["cross_attention_kwargs"]
    do_classifier_free_guidance = kwargs["do_classifier_free_guidance"]
    batch_size = kwargs["batch_size"]

    pipe._interrupt = False
    device = pipe._execution_device

    # Encode input prompt
    prompt_embeds, negative_prompt_embeds = encode_prompt(pipe, **kwargs)

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
        )

    # Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    return (
        latents,
        {
            "encoder_hidden_states": prompt_embeds,
            "timestep_cond": timestep_cond,
            "cross_attention_kwargs": cross_attention_kwargs,
            "added_cond_kwargs": added_cond_kwargs,
            "return_dict": False,
        },
    )


class UNet(nn.Module):
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        _unet_args: Dict[str, Any],
        guidance_scale: float,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self._unet_args = _unet_args
        self.guidance_scale = guidance_scale
    
    def to(self, device: Device) -> "UNet":
        self.pipe = self.pipe.to(device)
        return self

    def parameters(self) -> Iterator[Parameter]:
        return self.pipe.unet.parameters()

    def forward(self, latents: Tensor, _t: float) -> Tensor:
        do_classifier_free_guidance = (
            self.guidance_scale > 1 and self.pipe.unet.config.time_cond_proj_dim is None
        )
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.pipe.scheduler.scale_model_input(
            latent_model_input, _t
        )
        return self.pipe.unet(latent_model_input, _t, **self._unet_args)[0]


class SDv1(nn.Module):
    def __init__(
        self,
        version: str,
        scheduler: str,
        blur_nsfw: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = SDv1Config()

        self.pipe = load_pipeline(version, scheduler, **kwargs)
        self.device = self.pipe._execution_device
        self._unet = UNet(self.pipe, None, self.config.guidance_scale).to(self.device)

        self.blur_nsfw = blur_nsfw
        self.safety_checker = SafetyChecker(self.device)

        self.verbose = verbose
        self.desc = lambda prompt: f"\033[1m{prompt}\033[0m@SDv{version}_{scheduler}"

        self._seed: int = None
        self._timesteps: Tensor = self.config.timesteps
        self._num_inference_steps: int = self.config.num_inference_steps
        self._unet_args: Dict[str, Any] = None
        self._params: Dict[str, Any] = None

        self.noise_preds: List[Tensor] = []
        self.latents: List[Tensor] = []

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
        self._unet = self._unet.to(self.device)
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

    def make_latents(
        self,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Any]]:
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
        config = self.config.copy()
        config.update(**kwargs)
        self._params = load_params(self.pipe, config)

        self._timesteps, self._num_inference_steps = load_timesteps(
            pipe=self.pipe,
            **self._params,
        )
        latents, self._unet_args = prepare_unet_inputs(
            pipe=self.pipe,
            **self._params,
        )
        return latents

    def unet(self) -> Module:
        assert self._unet_args is not None and self._params is not None
        self._unet._unet_args = self._unet_args
        self._unet.guidance_scale = self._params["guidance_scale"]
        return self._unet

    @torch.no_grad()
    def forward(
        self,
        latents: Tensor,
        t: Optional[int] = 0,
    ) -> Tensor:
        self.noise_preds = []
        self.latents = []

        guidance_scale = self._params["guidance_scale"]
        eta = self._params["eta"]
        generator = self._params["generator"]
        guidance_rescale = self._params["guidance_rescale"]
        do_classifier_free_guidance = self._params["do_classifier_free_guidance"]
        unet = self.unet()

        for _t in tqdm.tqdm(
            self._timesteps[: self._num_inference_steps - t],
            desc=self.desc(self._params["prompt"]),
            disable=not self.verbose,
        ):
            if self.pipe._interrupt:
                continue

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latents, _t)
            self.noise_preds.append(noise_pred)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            # compute the previous noisy sample x_t -> x_t-1
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
            latents = self.pipe.scheduler.step(
                noise_pred, _t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            # TODO: XLA is not supported yet
            # if XLA_AVAILABLE:
            #     xm.mark_step()

            self.latents.append(latents)

        self.noise_preds = self.noise_preds[::-1]
        self.latents = self.latents[::-1]

        return latents

    @torch.no_grad()
    def encode_prompt(
        self,
        **kwargs,
    ) -> Tensor:
        return encode_prompt(self.pipe, **kwargs)

    @torch.no_grad()
    def encode_image(
        self,
        **kwargs,
    ) -> Tensor:
        return encode_image(self.pipe, **kwargs)

    @torch.no_grad()
    def decode_to_image(
        self,
        latents: Tensor,
    ) -> List[Image]:
        return decode_to_image(self.pipe, latents, self.config.generator)
