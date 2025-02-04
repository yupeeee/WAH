from torch import nn

from ....misc.typing import Module

__all__ = [
    "gelu_with_relu",
    "relu_with_gelu",
]


def gelu_with_relu(model: Module) -> Module:
    """Replace GELU activation with ReLU activation in a model.

    ### Args
        - `model` (Module): Model to replace GELU activation with ReLU activation

    ### Returns
        - `Module`: Model with GELU activation replaced with ReLU activation

    ### Example
    ```python
    >>> model = torchvision.models.vit_b_16()
    >>> model = gelu_with_relu(model)
    ```
    """
    for name, module in model.named_children():
        if isinstance(module, nn.GELU):
            relu = nn.ReLU(inplace=True)
            setattr(model, name, relu)
        else:
            gelu_with_relu(module)

    return model


def relu_with_gelu(model: Module) -> Module:
    """Replace ReLU activation with GELU activation in a model.

    ### Args
        - `model` (Module): Model to replace ReLU activation with GELU activation

    ### Returns
        - `Module`: Model with ReLU activation replaced with GELU activation

    ### Example
    ```python
    >>> model = torchvision.models.resnet18()
    >>> model = relu_with_gelu(model)
    ```
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            gelu = nn.GELU()
            setattr(model, name, gelu)
        else:
            relu_with_gelu(module)

    return model
