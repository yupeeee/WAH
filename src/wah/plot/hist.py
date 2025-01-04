import numpy as np
from matplotlib import cm

from ..typing import Axes, Figure, NDArray, Optional, PathCollection, Sequence, Tuple

__all__ = [
    "_hist2d",
    "_ridge",
]


def _hist(
    x: Sequence[float],
    num_bins: int,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> Tuple[NDArray, NDArray]:
    if xmin is None:
        xmin = min(x)
    if xmax is None:
        xmax = max(x)

    num_x = len(x)
    bins = np.linspace(xmin, xmax, num_bins)
    hist, bin_edges = np.histogram(x, bins)
    hist = hist / num_x

    return bins, hist


def _hist2d(
    fig: Figure,
    ax: Axes,
    x: Sequence[float],
    num_bins: int,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    *args,
    **kwargs,
) -> PathCollection:
    bins, hist = _hist(x, num_bins, xmin, xmax)
    plot: PathCollection = ax.plot(bins[:-1], hist, *args, **kwargs)

    return plot


def _ridge(
    fig: Figure,
    ax: Axes,
    xs: Sequence[Sequence[float]],
    num_bins: int,
    stride: Optional[int] = 1,
    overlap: Optional[float] = 0.01,
    alpha: Optional[float] = 0.8,
    cmap: Optional[str] = "viridis",
    *args,
    **kwargs,
) -> PathCollection:
    T = len(xs)
    colors = getattr(cm, cmap)(np.linspace(0, 1, T))
    ts = np.insert(np.arange(stride, T + stride, stride) - 1, 0, 0)

    for i, t in enumerate(reversed(ts)):
        # Get this timestep's data
        x = xs[t]

        # Compute histogram
        counts, bin_edges = np.histogram(x, bins=num_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Offset for the ridgeline
        # The bottom is "i * overlap" for all points
        y_offset = i * overlap

        # Plot
        ax.fill_between(
            bin_centers,
            y_offset,
            counts + y_offset,
            color=colors[t],
            alpha=alpha,
            zorder=t,
            *args,
            **kwargs,
        )
