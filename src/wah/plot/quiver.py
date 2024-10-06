import torch
from matplotlib import cm, colors

from ..typing import Axes, Figure, Iterable, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "QuiverPlot2D",
    "TrajPlot2D",
]


class QuiverPlot2D(Plot2D):
    """
    A class for creating 2D quiver plots using matplotlib, extending the `Plot2D` class.

    Inherits plot settings and customization from `Plot2D`, and adds functionality to visualize vector fields using quiver plots.

    ### Plot Components
    - Displays a vector field as arrows.
    - Supports optional color mapping for the vectors, with a color bar that can be customized.
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

        - `clabel` (str, optional): Label for the color bar.
        - `clim` (Tuple[float, float], optional): Minimum and maximum values for the color bar.
        - `cticks` (Tuple[float, float], optional): Positions for the color bar ticks.
        - `cticklabels` (Tuple[float, float], optional): Labels for the color bar ticks.
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
        vx: Iterable[float],
        vy: Iterable[float],
        scale: Optional[float] = 1.0,
        c: Optional[Iterable[float]] = None,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        """
        Plots a quiver plot for a vector field.

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `x` (Iterable[float]): X-coordinates of the vectors.
        - `y` (Iterable[float]): Y-coordinates of the vectors.
        - `vx` (Iterable[float]): X-components of the vectors.
        - `vy` (Iterable[float]): Y-components of the vectors.
        - `scale` (float, optional): Scaling factor for the vectors. Defaults to `1.0`.
        - `c` (Iterable[float], optional): Optional color data for the vectors.
        - `cmap` (str, optional): Colormap to use for the vector colors. Defaults to `"viridis"`.

        ### Plot Components
        - Vectors are displayed as arrows with direction and magnitude.
        - Color mapping can be applied to the vectors with an optional color bar.
        """
        if c is not None:
            # normalize c in range [cmin, cmax]
            if self.clim is not None:
                cmin, cmax = self.clim
            else:
                cmin, cmax = min(c), max(c)
            cnorm = colors.Normalize(cmin, cmax)
            colormap = getattr(cm, cmap)
            c = colormap(cnorm(c))

            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                color=c,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )
            cbar = fig.colorbar(plot, ax=ax)

            cbar.set_label(self.clabel)
            if self.cticks is not None:
                cbar.set_ticks(self.cticks)
                cbar.set_ticklabels(self.cticklabels)

        else:
            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )


class TrajPlot2D(QuiverPlot2D):
    """
    A class for plotting 2D trajectories using matplotlib quiver plots, extending `QuiverPlot2D`.

    Inherits functionality from `QuiverPlot2D` and customizes it for trajectory visualization.

    ### Plot Components
    - Trajectories are displayed as sequences of arrows representing motion.
    - Supports optional color mapping and color bar for additional data representation.
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

        - `clabel` (str, optional): Label for the color bar.
        - `clim` (Tuple[float, float], optional): Minimum and maximum values for the color bar.
        - `cticks` (Tuple[float, float], optional): Positions for the color bar ticks.
        - `cticklabels` (Tuple[float, float], optional): Labels for the color bar ticks.
        - `grid_alpha` (float, optional): Alpha (transparency) for the grid. Defaults to `0.0`.
        """
        super().__init__(
            figsize=figsize,
            fontsize=fontsize,
            title=title,
            xlabel=xlabel,
            xlim=xlim,
            xticks=xticks,
            xticklabels=xticklabels,
            ylabel=ylabel,
            ylim=ylim,
            yticks=yticks,
            yticklabels=yticklabels,
            clabel=clabel,
            clim=clim,
            cticks=cticks,
            cticklabels=cticklabels,
            grid_alpha=grid_alpha,
        )

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        traj: Tensor,
        scale: Optional[float] = 1.0,
        c: Optional[Iterable[float]] = None,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        """
        Plots a 2D trajectory using a quiver plot to represent the sequence of movement.'

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `traj` (Tensor): A tensor representing the trajectory, with shape `(N, 2)` for `N` points.
        - `scale` (float, optional): Scaling factor for the arrows. Defaults to `1.0`.
        - `c` (Iterable[float], optional): Optional color data for the arrows.
        - `cmap` (str, optional): Colormap to use for the arrow colors. Defaults to `"viridis"`.

        ### Plot Components
        - Trajectories are displayed as arrows showing direction and magnitude of motion.
        - Color mapping can be applied with an optional color bar.
        """
        arrows = torch.diff(traj, dim=0)
        x, y = traj[:-1, 0], traj[:-1, 1]
        vx, vy = arrows[:, 0], arrows[:, 1]

        if c is not None:
            # normalize c in range [cmin, cmax]
            if self.clim is not None:
                cmin, cmax = self.clim
            else:
                cmin, cmax = min(c), max(c)
            cnorm = colors.Normalize(cmin, cmax)
            colormap = getattr(cm, cmap)
            c = colormap(cnorm(c))

            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                color=c,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )
            cbar = fig.colorbar(plot, ax=ax)

            cbar.set_label(self.clabel)
            if self.cticks is not None:
                cbar.set_ticks(self.cticks)
                cbar.set_ticklabels(self.cticklabels)

        else:
            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )
