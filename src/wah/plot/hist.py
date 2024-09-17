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
    num_x = len(x)
    bins = np.linspace(x_min, x_max, num_bins)
    hist, bin_edges = np.histogram(x, bins)
    hist = hist / num_x

    return bins, hist


class HistPlot2D(Plot2D):
    """
    A class for creating 2D histogram plots with customizable settings.

    ### Attributes
    - `figsize (Optional[Tuple[float, float]])`: The size of the figure.
    - `fontsize (Optional[float])`: The font size for the plot text.
    - `title (Optional[str])`: The title of the plot.
    - `xlabel (Optional[str])`: The label for the x-axis.
    - `xlim (Optional[Tuple[float, float]])`: The limits for the x-axis.
    - `xticks (Optional[Iterable[float]])`: The tick values for the x-axis.
    - `xticklabels (Optional[Iterable[str]])`: The tick labels for the x-axis.
    - `ylabel (Optional[str])`: The label for the y-axis.
    - `ylim (Optional[Tuple[float, float]])`: The limits for the y-axis.
    - `yticks (Optional[Iterable[float]])`: The tick values for the y-axis.
    - `yticklabels (Optional[Iterable[str]])`: The tick labels for the y-axis.
    - `grid_alpha (Optional[float])`: The alpha transparency for the grid.

    ### Methods
    - `__init__(...)`: Initializes the HistPlot2D object with customizable settings.
    - `_plot(fig, ax, x, x_min, x_max, num_bins, *args, **kwargs) -> None`: Plots the histogram based on the provided data.
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
        - `figsize (Optional[Tuple[float, float]])`: The size of the figure. Defaults to None.
        - `fontsize (Optional[float])`: The font size for the plot text. Defaults to None.
        - `title (Optional[str])`: The title of the plot. Defaults to None.
        - `xlabel (Optional[str])`: The label for the x-axis. Defaults to None.
        - `xlim (Optional[Tuple[float, float]])`: The limits for the x-axis. Defaults to None.
        - `xticks (Optional[Iterable[float]])`: The tick values for the x-axis. Defaults to None.
        - `xticklabels (Optional[Iterable[str]])`: The tick labels for the x-axis. Defaults to None.
        - `ylabel (Optional[str])`: The label for the y-axis. Defaults to None.
        - `ylim (Optional[Tuple[float, float]])`: The limits for the y-axis. Defaults to None.
        - `yticks (Optional[Iterable[float]])`: The tick values for the y-axis. Defaults to None.
        - `yticklabels (Optional[Iterable[str]])`: The tick labels for the y-axis. Defaults to None.
        - `grid_alpha (Optional[float])`: The alpha transparency for the grid. Defaults to 0.0.
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
        Plots the histogram based on the provided data.

        ### Parameters
        - `fig (Figure)`: The Matplotlib figure object.
        - `ax (Axes)`: The Matplotlib axes object.
        - `x (Iterable[float])`: The data to compute the histogram for.
        - `x_min (float)`: The minimum value of the bins.
        - `x_max (float)`: The maximum value of the bins.
        - `num_bins (int)`: The number of bins.
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Returns
        - `None`

        ### Notes
        - This method generates a histogram plot of the data using Matplotlib.
        """
        bins, hist = _hist(x, x_min, x_max, num_bins)
        ax.plot(bins[:-1], hist, *args, **kwargs)
