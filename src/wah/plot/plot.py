from ..typing import Axes, Figure, PathCollection, Sequence

__all__ = [
    "_plot2d",
]


def _plot2d(
    fig: Figure,
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    *args,
    **kwargs,
) -> PathCollection:
    plot: PathCollection = ax.plot(x, y, *args, **kwargs)

    return plot
