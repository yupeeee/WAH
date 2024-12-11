from ..typing import Axes, Figure, Optional, PathCollection, Sequence

__all__ = [
    "_errorbar2d",
]


def _errorbar2d(
    fig: Figure,
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    yerr: Sequence[float],
    xerr: Optional[Sequence[float]] = None,
    *args,
    **kwargs,
) -> PathCollection:
    plot: PathCollection = ax.errorbar(x, y, yerr, xerr, *args, **kwargs)

    return plot
