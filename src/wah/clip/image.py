import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from ..misc.typing import CLIPOutput, Device, Image, List, Union

__all__ = [
    "ImageEncoder",
]


class ImageEncoder:
    def __init__(
        self,
        model_name: str,
        device: Device = "cpu",
    ) -> None:
        self.device = device

        self.preprocess = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)

    def to(self, device: Device) -> "ImageEncoder":
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(
        self,
        image: Union[Image, List[Image]],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> CLIPOutput:
        inputs = self.preprocess(
            image,
            return_tensors="pt",
        )
        with torch.no_grad():
            output: CLIPOutput = self.model(
                **inputs.to(self.device),
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                **kwargs,
            )
        return output
