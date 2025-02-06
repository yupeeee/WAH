import torch
import tqdm
from torch import nn

from ..misc import mods as _mods
from ..misc.typing import Device, List, Module, Tensor, Tuple

__all__ = [
    "RecursionWrapper",
]


class RecursionWrapper(nn.Module):
    """Wrapper for performing recursive feature injection between model layers.

    ### Args
        - `model` (Module): Model to wrap
        - `out_layer` (str): Layer to extract features from
        - `in_layer` (str): Layer to inject features into
        - `num_iter` (int): Number of recursive iterations. Defaults to `1`.
        - `device` (Device): Device to run model on. Defaults to `"cpu"`.

    ### Returns
        - `injections` (List[Tensor]): List of injection tensors
        - `outputs` (List[Tensor]): List of model outputs

    ### Attributes
        - `model` (Module): Wrapped model
        - `out_layer` (str): Layer to extract features from
        - `in_layer` (str): Layer to inject features into
        - `num_iter` (int): Number of recursive iterations
        - `device` (Device): Device model is running on
        - `injection` (Tensor): Current injection tensor
        - `injections` (List[Tensor]): List of injection tensors
        - `outputs` (List[Tensor]): List of model outputs

    ### Example
    ```python
    >>> model = wah.classification.models.load("resnet18", weights="auto")
    >>> model = RecursionWrapper(
    ...     model=model,
    ...     out_layer="layer4.1.act2",
    ...     in_layer="layer4.1.conv1",
    ...     num_iter=10,
    ... )
    >>> x = torch.randn(1, 3, 224, 224)
    >>> injections, outputs = model(x)
    ```
    """

    def __init__(
        self,
        model: Module,
        out_layer: str,
        in_layer: str,
        num_iter: int = 1,
        device: Device = "cpu",
    ) -> None:
        """
        - `model` (Module): Model to wrap
        - `out_layer` (str): Layer to extract features from
        - `in_layer` (str): Layer to inject features into
        - `num_iter` (int): Number of recursive iterations. Defaults to `1`.
        - `device` (Device): Device to run model on. Defaults to `"cpu"`.
        """
        super().__init__()
        self.model = model
        self.out_layer = out_layer
        self.in_layer = in_layer
        self.num_iter = num_iter
        self.device = device
        self.injection: Tensor = None
        self.injections: List[Tensor] = []
        self.outputs: List[Tensor] = []

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        self.model = self.model.to(self.device)
        x = x.to(self.device)
        for _ in tqdm.trange(self.num_iter, desc="Recursive Injection"):
            out_hook = _mods.getmod(self.model, self.out_layer).register_forward_hook(
                self._out_hook
            )
            with torch.no_grad():
                y = self.model(x)
            out_hook.remove()
            self.injections.append(self.injection)
            # Inject
            in_hook = _mods.getmod(self.model, self.in_layer).register_forward_pre_hook(
                self._in_hook
            )
            with torch.no_grad():
                y = self.model(x)
            in_hook.remove()
            self.outputs.append(y)

        return self.injections, self.outputs

    def _out_hook(self, module: Module, input: Tensor, output: Tensor) -> Tensor:
        self.injection = output
        return output

    def _in_hook(self, module: Module, input: Tensor) -> Tensor:
        assert self.injection is not None
        self.injection = self.injection.reshape(
            len(self.injection), *input[0].shape[1:]
        )
        output = module.forward(*(self.injection,))
        self.injection = None
        return output
