from ..typing import Axes, Figure, NDArray, PathCollection, Tensor, Union

__all__ = [
    "_image",
]


def _image(
    fig: Figure,
    ax: Axes,
    x: Union[NDArray, Tensor],
    *args,
    **kwargs,
) -> PathCollection:
    assert x.dim() in [
        2,
        3,
    ], f"Expected 2D (grayscale) or 3D (RGB) image, got {x.dim()}D input"

    plot: PathCollection = ax.imshow(x, *args, **kwargs)

    return plot
