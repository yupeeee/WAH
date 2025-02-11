import torch
import tqdm
from torch import nn

from ..misc import mods as _mods
from ..misc.typing import Device, List, Module, Optional, Sequence, Tensor, Tuple

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
        - `permute` (Sequence[int], optional): Permutation to apply to the injection tensor. Defaults to `None`.
        - `device` (Device): Device to run model on. Defaults to `"cpu"`.
        - `msg` (str, optional): Message to display in progress bar. If None, no progress bar will be shown. Defaults to `None`.

    ### Returns
        - `injections` (List[Tensor]): List of injection tensors
        - `outputs` (List[Tensor]): List of model outputs

    ### Attributes
        - `model` (Module): Wrapped model
        - `out_layer` (str): Layer to extract features from
        - `in_layer` (str): Layer to inject features into
        - `num_iter` (int): Number of recursive iterations
        - `permute` (Sequence[int], optional): Permutation to apply to the injection tensor. Defaults to `None`.
        - `device` (Device): Device model is running on
        - `msg` (str, optional): Message to display in progress bar
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
        permute: Sequence[int] = None,
        device: Device = "cpu",
        msg: Optional[str] = None,
    ) -> None:
        """
        - `model` (Module): Model to wrap
        - `out_layer` (str): Layer to extract features from
        - `in_layer` (str): Layer to inject features into
        - `num_iter` (int): Number of recursive iterations. Defaults to `1`.
        - `permute` (Sequence[int], optional): Permutation to apply to the injection tensor. Defaults to `None`.
        - `device` (Device): Device to run model on. Defaults to `"cpu"`.
        - `msg` (str, optional): Message to display in progress bar. If None, no progress bar will be shown. Defaults to `None`.
        """
        super().__init__()
        self.model = model
        self.out_layer = out_layer
        self.in_layer = in_layer
        self.num_iter = num_iter
        self.permute = permute
        self.device = device
        self.msg = msg
        self.injection: Tensor = None
        self.injections: List[Tensor] = []
        self.outputs: List[Tensor] = []

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        self.model = self.model.to(self.device)
        x = x.to(self.device)

        # Fetch initial injection
        init_hook = _mods.getmod(self.model, self.out_layer).register_forward_hook(
            self._init_hook
        )
        with torch.no_grad():
            y = self.model(x)
        init_hook.remove()
        self.injections.append(self.injection.clone().cpu())
        self.outputs.append(y.clone().cpu())

        # Recursively inject
        loop_hook = _mods.getmod(self.model, self.in_layer).register_forward_pre_hook(
            self._loop_hook
        )
        for _ in tqdm.trange(self.num_iter, desc=self.msg, disable=self.msg is None):
            with torch.no_grad():
                y = self.model(x)
            self.injections.append(self.injection.clone().cpu())
            self.outputs.append(y.clone().cpu())
        loop_hook.remove()
        return self.injections, self.outputs

    def _init_hook(self, module: Module, input: Tensor, output: Tensor) -> Tensor:
        self.injection = output
        return output

    def _loop_hook(self, module: Module, input: Tensor) -> Tensor:
        assert self.injection is not None
        if self.permute is not None:
            self.injection = self.injection.permute(*self.permute)
        self.injection = self.injection.reshape(
            len(self.injection), *input[0].shape[1:]
        )
        output = module.forward(*(self.injection,))
        self.injection = output
        return output
