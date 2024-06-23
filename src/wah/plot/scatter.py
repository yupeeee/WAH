from ..typing import (
    Axes,
    Iterable,
    Optional,
    Tuple,
)
from .base import Plot2D

__all__ = [
    "ScatterPlot2D",
]


class ScatterPlot2D(Plot2D):
    """
    A class for creating 2D scatter plots with customizable settings.

    ### Methods
    - `plot`:
      Creates the plot with the specified settings.
    - `show`:
      Displays the plot.
    - `save`:
      Saves the plot to the specified path with optional settings.
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
        ax: Axes,
        x: Iterable[float],
        y: Iterable[float],
        *args,
        **kwargs,
    ) -> None:
        ax.scatter(x, y, *args, **kwargs)
