from typing import List, Tuple, Union

import torch
import torchvision.transforms as T
from PIL.Image import Image
from transformers import CLIPModel, CLIPTokenizer

__all__ = [
    "CLIP",
]

# OpenAI CLIP checkpoints on the Hub use the openai/clip-* id scheme. We validate
# locally so construction does not call list_models() (no network; safe for DDP).
_KNOWN_OPENAI_CLIP_IDS: Tuple[str, ...] = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
    "openai/clip-rn50",
    "openai/clip-rn101",
)


def _is_openai_clip_model_id(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith("openai/clip-")


def _is_text(x) -> bool:
    """Returns True if x is a string or list of strings (Union[str, List[str]]), False otherwise"""
    return isinstance(x, (str, list)) and (
        isinstance(x, str) or all(isinstance(s, str) for s in x)
    )


class CLIP:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if not _is_openai_clip_model_id(model_id):
            raise ValueError(
                f"Model '{model_id}' must be an OpenAI CLIP Hub id (prefix 'openai/clip-'). "
                f"Examples: {', '.join(_KNOWN_OPENAI_CLIP_IDS)}"
            )

        self.model_id = model_id
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.clip_model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.clip_model.eval()

        # Text preprocessing is shared across OpenAI CLIP models
        self.text_preprocess = CLIPTokenizer.from_pretrained(self.model_id)

        # Image preprocessing: only image size varies between OpenAI CLIP variants.
        image_size = self.clip_model.vision_model.config.image_size
        self.image_preprocess = T.Compose(
            [
                T.Resize(
                    image_size,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),  # OpenAI stats
            ]
        )

    def to(self, device: torch.device) -> "CLIP":
        self.device = device
        self.clip_model = self.clip_model.to(device)
        return self

    def _embed_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        inputs = self.text_preprocess(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"]
            )

        return embeddings

    def _embed_image(self, image: Union[Image, List[Image]]) -> torch.Tensor:
        if isinstance(image, Image):
            image = [image]
        image = [self.image_preprocess(img) for img in image]
        image = torch.stack(image, dim=0)

        image = image.to(self.device)

        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(pixel_values=image)

        return embeddings

    def __call__(
        self, input: Union[str, List[str], Image, List[Image]]
    ) -> torch.Tensor:
        if _is_text(input):
            return self._embed_text(input)
        else:
            return self._embed_image(input)
