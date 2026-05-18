import torch
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = [
    "_get_prompt_embeds",
]

logger = logging.get_logger(__name__)


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/textual_inversion.py#L150
def __maybe_convert_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:  # noqa: F821
    r"""
    Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
    to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
    is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
    inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

    Parameters:
        prompt (`str`):
            The prompt to guide the image generation.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer responsible for encoding the prompt into input tokens.

    Returns:
        `str`: The converted prompt
    """
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1

            prompt = prompt.replace(token, replacement)

    return prompt


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/textual_inversion.py#L123
def _maybe_convert_prompt(
    prompt: str | list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> str | list[str]:  # noqa: F821
    r"""
    Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
    be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
    inversion token or if the textual inversion token is a single vector, the input prompt is returned.

    Parameters:
        prompt (`str` or list of `str`):
            The prompt or prompts to guide the image generation.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer responsible for encoding the prompt into input tokens.

    Returns:
        `str` or list of `str`: The converted prompt
    """
    if not isinstance(prompt, list):
        prompts = [prompt]
    else:
        prompts = prompt

    prompts = [__maybe_convert_prompt(p, tokenizer) for p in prompts]

    if not isinstance(prompt, list):
        return prompts[0]

    return prompts


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L332
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L218
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L267
def _get_prompt_embeds(
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
    prompt: str | list[str],
    max_sequence_length: int,
    use_pooled_output: bool,  # False for _get_t5_prompt_embeds, True for _get_clip_prompt_embeds
    num_images_per_prompt: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    lora_scale: float | None = None,
    clip_skip: int | None = None,
) -> Tensor:
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to tokenize the prompt.
        text_encoder (`PreTrainedModel`):
            The text encoder to encode the prompt.
        prompt (`str` or `list[str]`):
            The prompt to be encoded.
        max_sequence_length (`int`):
            The maximum sequence length allowed by the text encoder.
        num_images_per_prompt (`int`):
            The number of images that should be generated per prompt.
        device: (`torch.device`):
            The device to put the prompt embeddings on.
        dtype (`torch.dtype`):
            The data type for the text embeddings.
        lora_scale (`float`, *optional*):
            A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if (
        lora_scale is not None
    ):  # and isinstance(self, StableDiffusionLoraLoaderMixin): <- removed since can be manually triggered via lora_scale
        # dynamically adjust the LoRA scale
        if not USE_PEFT_BACKEND:
            adjust_lora_scale_text_encoder(text_encoder, lora_scale)
        else:
            scale_lora_layers(text_encoder, lora_scale)

    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # textual inversion: process multi-vector tokens if necessary
    # if isinstance(self, TextualInversionLoaderMixin): <- removed since pipe is almost surely to be an instance of TextualInversionLoaderMixin
    prompt = _maybe_convert_prompt(prompt, tokenizer)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    # _get_t5_prompt_embeds
    if use_pooled_output:
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=False,
        )
        prompt_embeds = prompt_embeds.pooler_output

    # _get_clip_prompt_embeds
    else:
        if (
            hasattr(text_encoder.config, "use_attention_mask")
            and text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    if use_pooled_output:
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    else:
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        if USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(text_encoder, lora_scale)

    return prompt_embeds
