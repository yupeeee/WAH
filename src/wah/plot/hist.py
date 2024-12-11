import numpy as np

from ..typing import (
    Axes,
    Figure,
    List,
    NDArray,
    Optional,
    PathCollection,
    Sequence,
    Tuple,
)

__all__ = [
    "_hist2d",
    "_hist3d",
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
    num_bins: float,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    *args,
    **kwargs,
) -> PathCollection:
    bins, hist = _hist(x, num_bins, xmin, xmax)
    plot: PathCollection = ax.plot(bins[:-1], hist, *args, **kwargs)

    return plot


def _hist3d(
    fig: Figure,
    ax: Axes,
    xs: Sequence[Sequence[float]],
    num_bins: float,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    t: Optional[Sequence[float]] = None,
    *args,
    **kwargs,
) -> PathCollection:
    if t is None:
        t = np.arange(len(xs))
    else:
        assert len(t) == len(
            xs
        ), f"Length of timestep (len(t)) must match the number of trials for x (len(xs)), got len(t)={len(t)} and len(xs)={len(xs)}."

    bins_per_t: List[NDArray] = []
    hists: List[NDArray] = []
    for i in range(len(t)):
        bins, hist = _hist(xs[i], num_bins, xmin, xmax)
        bins_per_t.append(bins)
        hists.append(hist)

    for i, (bins, hist) in enumerate(zip(bins_per_t, hists)):
        ax.plot(bins[:-1], hist + i, *args, **kwargs)
        ax.fill_between(bins[:-1], i, hist + i, *args, **kwargs)
