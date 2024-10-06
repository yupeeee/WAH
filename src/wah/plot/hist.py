import numpy as np

from ..typing import Axes, Figure, Iterable, Optional, Tuple
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
    Computes the histogram data for the given input `x`.

    ### Parameters
    - `x` (Iterable[float]): The input data to compute the histogram for.
    - `x_min` (float): The minimum value of the range for the histogram.
    - `x_max` (float): The maximum value of the range for the histogram.
    - `num_bins` (int): The number of bins to divide the data into.

    ### Returns
    - `Tuple[Iterable[float], Iterable[float]]`: A tuple containing the bins and the normalized histogram values.
    """
    num_x = len(x)
    bins = np.linspace(x_min, x_max, num_bins)
    hist, bin_edges = np.histogram(x, bins)
    hist = hist / num_x

    return bins, hist


class HistPlot2D(Plot2D):
    """
    A class for creating 2D histogram plots using matplotlib, extending the `Plot2D` class.

    Inherits plot settings and customization from `Plot2D`, and adds functionality to plot histograms.

    ### Plot Components
    - The x-values are represented as bins.
    - The y-values represent the normalized frequency of the data.
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
        """
        - `figsize` (Tuple[float, float], optional): Figure size.
        - `fontsize` (float, optional): Font size for the plot text.
        - `title` (str, optional): Title of the plot.

        - `xlabel` (str, optional): X-axis label.
        - `xlim` (Tuple[float, float], optional): X-axis limits.
        - `xticks` (Iterable[float], optional): X-axis tick positions.
        - `xticklabels` (Iterable[str], optional): X-axis tick labels.

        - `ylabel` (str, optional): Y-axis label.
        - `ylim` (Tuple[float, float], optional): Y-axis limits.
        - `yticks` (Iterable[float], optional): Y-axis tick positions.
        - `yticklabels` (Iterable[str], optional): Y-axis tick labels.

        - `grid_alpha` (float, optional): Alpha (transparency) for the grid. Defaults to `0.0`.
        """
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
        """
        Plots a histogram based on the input data `x`.

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `x` (Iterable[float]): The input data to create the histogram from.
        - `x_min` (float): The minimum value for the bins.
        - `x_max` (float): The maximum value for the bins.
        - `num_bins` (int): The number of bins to divide the data into.

        ### Plot Components
        - The x-values are represented as bins.
        - The y-values represent the normalized frequency of the data.
        """
        bins, hist = _hist(x, x_min, x_max, num_bins)
        ax.plot(bins[:-1], hist, *args, **kwargs)
