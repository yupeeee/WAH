from torch import nn

from ....typing import Module

__all__ = [
    "replace_gelu_with_relu",
    "replace_relu_with_gelu",
]


def replace_gelu_with_relu(model: Module) -> Module:
    for name, module in model.named_children():
        if isinstance(module, nn.GELU):
            relu = nn.ReLU(inplace=True)
            setattr(model, name, relu)
        else:
            replace_gelu_with_relu(module)

    return model


def replace_relu_with_gelu(model: Module) -> Module:
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            gelu = nn.GELU()
            setattr(model, name, gelu)
        else:
            replace_relu_with_gelu(module)

    return model
