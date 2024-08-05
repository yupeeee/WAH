import numpy as np

from ..typing import (
    Axes,
    Figure,
    Iterable,
    Optional,
    Tuple,
)
from .base import Plot2D

__all__ = [
    "HistPlot2D",
]


def _hist(
    x: Iterable[float],
    x_min: float,
    x_max: float,
    num_bins: int,
) -> Tuple[Iterable[float], Iterable[float]]:
    """
    Computes the histogram of the given data.

    ### Parameters
    - `x (Iterable[float])`: The data to compute the histogram for.
    - `x_min (float)`: The minimum value of the bins.
    - `x_max (float)`: The maximum value of the bins.
    - `num_bins (int)`: The number of bins.

    ### Returns
    - `Tuple[Iterable[float], Iterable[float]]`: A tuple containing the bin edges and the normalized histogram values.

    ### Notes
    - This function computes the histogram of the given data and normalizes it by the total number of data points.
    """
    num_x = len(x)
    bins = np.linspace(x_min, x_max, num_bins)
    hist, bin_edges = np.histogram(x, bins)
    hist = hist / num_x

    return bins, hist


class HistPlot2D(Plot2D):
    """
    A class for creating 2D histogram plots with customizable settings.

    ### Methods
    - `plot`: Creates the plot with the specified settings.
    - `show`: Displays the plot.
    - `save`: Saves the plot to the specified path with optional settings.
    """

    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        xticks: Optional[Iterable[float]] = None,
        xticklabels: Optional[Iterable[str]] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        yticks: Optional[Iterable[float]] = None,
        yticklabels: Optional[Iterable[str]] = None,
        grid_alpha: Optional[float] = 0.0,
    ) -> None:
        super().__init__(
            figsize,
            fontsize,
            title,
            xlabel,
            xlim,
            xticks,
            xticklabels,
            ylabel,
            ylim,
            yticks,
            yticklabels,
            grid_alpha,
        )

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        x: Iterable[float],
        x_min: float,
        x_max: float,
        num_bins: int,
        *args,
        **kwargs,
    ) -> None:
        bins, hist = _hist(x, x_min, x_max, num_bins)
        ax.plot(bins[:-1], hist, *args, **kwargs)
