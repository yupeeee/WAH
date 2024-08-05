import numpy as np
from scipy.stats import gaussian_kde

from ..typing import (
    Axes,
    Figure,
    Iterable,
    NDArray,
    Optional,
    Tensor,
    Tuple,
)
from .base import Plot2D


__all__ = [
    "ScatterPlot2D",
    "DensityScatterPlot2D",
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
        clabel: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        cticks: Optional[Tuple[float, float]] = None,
        cticklabels: Optional[Tuple[float, float]] = None,
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
        self.clabel = clabel
        self.clim = clim
        self.cticks = cticks
        self.cticklabels = (
            [str(c) for c in cticks]
            if cticks is not None and cticklabels is None
            else cticklabels
        )

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        x: Iterable[float],
        y: Iterable[float],
        c: Optional[Iterable[float]] = None,
        *args,
        **kwargs,
    ) -> None:
        if c is not None:
            cmin, cmax = self.clim
            plot = ax.scatter(x, y, c=c, vmin=cmin, vmax=cmax, *args, **kwargs)
            cbar = fig.colorbar(plot, ax=ax)
            cbar.set_label(self.clabel)
            cbar.set_ticks(self.cticks)
            cbar.set_ticklabels(self.cticklabels)
        else:
            ax.scatter(x, y, *args, **kwargs)


class DensityScatterPlot2D(Plot2D):
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
        fig: Figure,
        ax: Axes,
        x: Iterable[float],
        y: Iterable[float],
        *args,
        **kwargs,
    ) -> None:
        if isinstance(x, Tensor):
            x = x.numpy()
        else:
            x = np.array(x)

        if isinstance(y, Tensor):
            y = y.numpy()
        else:
            y = np.array(y)

        xy = np.vstack([x, y])
        z: NDArray = gaussian_kde(xy)(xy)
        # z = z - min(z)
        # z = z / max(z)

        indices = z.argsort()
        x, y, z = x[indices], y[indices], z[indices]

        plot = ax.scatter(
            x,
            y,
            c=z,
            # vmin=0,
            # vmax=1,
            *args,
            **kwargs,
        )
        fig.colorbar(plot, ax=ax)
