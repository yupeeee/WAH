import torch

from ..typing import Axes, Figure, Iterable, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "MatShow2D",
]


class MatShow2D(Plot2D):
    """
    A class for creating 2D matrix plots using `matshow` in matplotlib, extending the `Plot2D` class.

    Inherits plot settings and customization from `Plot2D`, and adds functionality to plot matrices with color bars.

    ### Plot Components
    - The matrix is displayed as a heatmap with color mapping.
    - An optional color bar can be added with customizable ticks and labels.
    """

    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        xticks: Optional[Iterable[float]] = None,
        xticklabels: Optional[Iterable[str]] = None,
        ylabel: Optional[str] = None,
        yticks: Optional[Iterable[float]] = None,
        yticklabels: Optional[Iterable[str]] = None,
        clabel: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        cticks: Optional[Tuple[float, float]] = None,
        cticklabels: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        - `figsize` (Tuple[float, float], optional): Figure size.
        - `fontsize` (float, optional): Font size for the plot text.
        - `title` (str, optional): Title of the plot.

        - `xlabel` (str, optional): X-axis label.
        - `xticks` (Iterable[float], optional): X-axis tick positions.
        - `xticklabels` (Iterable[str], optional): X-axis tick labels.

        - `ylabel` (str, optional): Y-axis label.
        - `yticks` (Iterable[float], optional): Y-axis tick positions.
        - `yticklabels` (Iterable[str], optional): Y-axis tick labels.

        - `clabel` (str, optional): Label for the color bar.
        - `clim` (Tuple[float, float], optional): Minimum and maximum limits for the color bar.
        - `cticks` (Tuple[float, float], optional): Tick positions for the color bar.
        - `cticklabels` (Tuple[float, float], optional): Labels for the color bar ticks.
        """
        super().__init__(
            figsize=figsize,
            fontsize=fontsize,
            title=title,
            xlabel=xlabel,
            xlim=None,
            xticks=xticks,
            xticklabels=xticklabels,
            ylabel=ylabel,
            ylim=None,
            yticks=yticks,
            yticklabels=yticklabels,
            grid_alpha=0.0,
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
        mat: Tensor,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        """
        Plots a 2D matrix as a heatmap using `matshow`.

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `mat` (Tensor): A 2D tensor representing the matrix to plot.
        - `cmap` (str, optional): Colormap to use for the matrix plot. Defaults to `"viridis"`.

        ### Plot Components
        - The matrix is displayed as a heatmap with color mapping.
        - An optional color bar is added with customizable ticks and labels.
        """
        self.xlim = (0, mat.size(0) - 1)
        self.ylim = (0, mat.size(1) - 1)

        if self.clim is not None:
            cmin, cmax = self.clim
        else:
            cmin, cmax = torch.min(mat), torch.max(mat)
        plot = ax.matshow(mat, cmap=cmap, vmin=cmin, vmax=cmax, *args, **kwargs)
        cbar = fig.colorbar(plot, ax=ax)

        cbar.set_label(self.clabel)
        if self.cticks is not None:
            cbar.set_ticks(self.cticks)
            cbar.set_ticklabels(self.cticklabels)
