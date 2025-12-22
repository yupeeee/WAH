from typing import List, Union

import torch
import torchvision.transforms as T
from huggingface_hub import list_models
from PIL.Image import Image
from transformers import CLIPModel, CLIPTokenizer

__all__ = [
    "CLIP",
]


def available_models():
    return [model.modelId for model in list_models(filter="clip", author="openai")]


def is_text(x):
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
        assert (
            model_id in available_models()
        ), f"Model '{model_id}' is not supported. Available models: {', '.join(available_models())}"

        self.model_id = model_id
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.clip_model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.clip_model.eval()

        self.text_preprocess = CLIPTokenizer.from_pretrained(self.model_id)
        self.image_preprocess = T.Compose(
            [
                T.Resize(
                    224,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.CenterCrop(224),
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
        if is_text(input):
            return self._embed_text(input)
        else:
            return self._embed_image(input)
