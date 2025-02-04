import torch
from torch import nn

from ...misc.mods import getattrs, getmod
from ...misc.typing import Dict, List, Module, RemovableHandle, Tensor, Union

__all__ = [
    "FeatureExtractor",
]


class FeatureExtractor(nn.Module):
    """Feature extractor for PyTorch models.

    ### Args
        - `model` (Module): PyTorch model to extract features from
        - `attrs` (List[str], optional): List of attribute names to extract features from.
          If None, extracts from all modules. Defaults to None.
        - `penultimate_only` (bool, optional): Whether to only return features from the penultimate layer.
          Defaults to False.
        - `extract_inputs` (bool, optional): Whether to extract input features instead of output features.
          Defaults to False.

    ### Attributes
        - `model` (Module): PyTorch model to extract features from
        - `attrs` (List[str]): List of attribute names to extract features from
        - `penultimate_only` (bool): Whether to only return features from the penultimate layer
        - `extract_inputs` (bool): Whether to extract input features instead of output features

    ### Example
    ```python
    >>> import torch
    >>> from collections import OrderedDict
    >>> model = torch.nn.Sequential(OrderedDict([
    ...     ('conv', torch.nn.Conv2d(3, 64, 3)),
    ...     ('relu', torch.nn.ReLU()),
    ...     ('pool', torch.nn.MaxPool2d(2))
    ... ]))
    >>> extractor = FeatureExtractor(model)
    >>> x = torch.randn(1, 3, 32, 32)  # Create sample input
    >>> features = extractor(x)  # Extract features from all layers
    >>> print(features.keys())  # View available feature maps
    dict_keys(['conv', 'relu', 'pool'])
    >>> print(features['conv'].shape)  # View shape of conv layer features
    torch.Size([1, 64, 30, 30])
    ```
    """

    def __init__(
        self,
        model: Module,
        attrs: List[str] = None,
        penultimate_only: bool = False,
        extract_inputs: bool = False,
    ) -> None:
        """
        - `model` (Module): PyTorch model to extract features from
        - `attrs` (List[str], optional): List of attribute names to extract features from.
          If None, extracts from all modules. Defaults to None.
        - `penultimate_only` (bool, optional): Whether to only return features from the penultimate layer.
          Defaults to False.
        - `extract_inputs` (bool, optional): Whether to extract input features instead of output features.
          Defaults to False.
        """
        super().__init__()
        self.model = model
        self.attrs = attrs if attrs is not None else getattrs(model)
        self.penultimate_only = penultimate_only
        self.extract_inputs = extract_inputs

    def forward(self, x: Tensor) -> Union[Dict[str, Tensor], Tensor]:
        hooks: List[RemovableHandle] = []
        features: List[Tensor] = []

        def hook_fn(module, input, output):
            if not self.extract_inputs:
                features.append(output)
            else:
                features.append(input)

        for attr in self.attrs:
            hook_handle: RemovableHandle = getmod(
                self.model, attr
            ).register_forward_hook(hook_fn)
            hooks.append(hook_handle)
        with torch.no_grad():
            _ = self.model(x)
        for hook_handle in hooks:
            hook_handle.remove()
        features: Dict[str, Tensor] = dict(
            (self.attrs[i], features[i]) for i in range(len(self.attrs))
        )
        if self.penultimate_only:
            features = {self.attrs[-2]: features[self.attrs[-2]]}
        if len(features.keys()) == 1:
            features = list(features.values())[0]
        return features
