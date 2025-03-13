import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ..misc.typing import CLIPOutput, Device, List, Union

__all__ = [
    "TextEncoder",
]


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        device: Device = "cpu",
    ) -> None:
        self.device = device

        self.preprocess = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)

    def to(self, device: Device) -> "TextEncoder":
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(
        self,
        text: Union[str, List[str]],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> CLIPOutput:
        inputs = self.preprocess(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
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
