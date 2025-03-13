from ..misc.typing import CLIPOutput, Device, List, Tensor, Union
from .image import ImageEncoder
from .text import TextEncoder
from .utils import available_models

__all__ = [
    "CLIP",
]


def is_text(x):
    """Returns True if x is a string or list of strings (Union[str, List[str]]), False otherwise"""
    return isinstance(x, (str, list)) and (
        isinstance(x, str) or all(isinstance(s, str) for s in x)
    )


class CLIP:
    def __init__(self, model_name: str, device: Device = "cpu"):
        if model_name not in available_models():
            raise ValueError(
                f"Model '{model_name}' is not supported. Available models: {', '.join(available_models())}"
            )
        self.model_name = model_name
        self.text_encoder = TextEncoder(model_name, device)
        self.image_encoder = ImageEncoder(model_name, device)
        self.device = device

    def to(self, device: Device) -> "CLIP":
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.image_encoder = self.image_encoder.to(device)
        return self

    def encode(
        self,
        x,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> CLIPOutput:
        if is_text(x):
            return self.text_encoder(
                x,
                output_hidden_states,
                output_attentions,
                **kwargs,
            )
        else:
            return self.image_encoder(
                x,
                output_hidden_states,
                output_attentions,
                **kwargs,
            )

    def normalize(
        self,
        hidden_states: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        if isinstance(hidden_states, Tensor):
            return hidden_states / hidden_states.norm(dim=-1, keepdim=True)
        else:
            return [self.normalize(h) for h in hidden_states]
