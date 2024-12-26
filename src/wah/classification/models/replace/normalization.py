from timm.layers import LayerNorm2d
from torch import nn

from ....module import get_module_name
from ....typing import Module, Optional, Sequence, Tensor

__all__ = [
    "replace_bn_with_ln",
    "replace_ln_with_bn",
]


class PermuteWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        dims: Sequence[int],
        unsqueeze_dim: Optional[int] = None,
    ):
        super().__init__()
        self.module = module
        self.dims = dims
        self.inverse_dims = sorted(range(len(self.dims)), key=lambda i: self.dims[i])
        self.unsqueeze_dim = unsqueeze_dim

    def forward(self, x: Tensor) -> Tensor:
        unsqueeze_flag = False

        if len(x.shape) == 3 and self.unsqueeze_dim is not None:
            unsqueeze_flag = True
            x = x.unsqueeze(dim=self.unsqueeze_dim)

        x = (
            self.module(x.permute(self.dims).contiguous())
            .permute(self.inverse_dims)
            .contiguous()
        )

        if unsqueeze_flag:
            x = x.squeeze(dim=self.unsqueeze_dim)

        return x


def replace_bn_with_ln(model: Module) -> Module:
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            eps = module.eps
            affine = module.affine

            # layer_norm = nn.LayerNorm(
            #     normalized_shape=(num_features,),
            #     eps=eps,
            #     elementwise_affine=affine,
            # )
            # setattr(model, name, PermuteWrapper(layer_norm, (0, 2, 3, 1), None))
            layer_norm = LayerNorm2d(
                num_channels=num_features,
                eps=eps,
                affine=affine,
            )
            setattr(model, name, layer_norm)
        else:
            replace_bn_with_ln(module)

    return model


def replace_ln_with_bn(model: Module) -> Module:
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine
            assert len(normalized_shape) == 1

            batch_norm = nn.BatchNorm2d(
                num_features=normalized_shape[0],
                eps=eps,
                affine=elementwise_affine,
            )
            if get_module_name(module) == "LayerNorm2d":
                setattr(model, name, batch_norm)
            else:
                setattr(model, name, PermuteWrapper(batch_norm, (0, 3, 1, 2), 1))
        else:
            replace_ln_with_bn(module)

    return model
