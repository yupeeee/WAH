import torch

from ..typing import Axes, Figure, Iterable, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "GridPlot2D",
    "ScatterPlot2D",
]


class ScatterPlot2D(Plot2D):
    """
    A class for creating 2D scatter plots using matplotlib, extending the `Plot2D` class.

    Inherits plot settings and customization from `Plot2D`, and adds functionality for plotting scatter plots with optional color bars and identity lines.

    ### Plot Components
    - Data points are displayed as individual scatter points.
    - Optionally, a color bar can be added to represent an additional dimension of data.
    - An optional identity line can be plotted, showing where `x = y`.
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
        c: Optional[Iterable[float]] = None,
        plot_identity_line: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Plots a 2D scatter plot.

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `x` (Iterable[float]): X-coordinates of the scatter points.
        - `y` (Iterable[float]): Y-coordinates of the scatter points.
        - `c` (Iterable[float], optional): Optional color data for the points.
        - `plot_identity_line` (bool, optional): If `True`, plots a dashed identity line (`x = y`). Defaults to `False`.

        ### Plot Components
        - Data points are displayed as scatter points.
        - A color bar can be added for color-coded data representation.
        - Optionally, an identity line can be plotted to compare `x` and `y` values.
        """
        if c is not None:
            if self.clim is not None:
                cmin, cmax = self.clim
            else:
                cmin, cmax = min(c), max(c)
            plot = ax.scatter(x, y, c=c, vmin=cmin, vmax=cmax, *args, **kwargs)
            cbar = fig.colorbar(plot, ax=ax)

            cbar.set_label(self.clabel)
            if self.cticks is not None:
                cbar.set_ticks(self.cticks)
                cbar.set_ticklabels(self.cticklabels)

        else:
            ax.scatter(x, y, *args, **kwargs)

        if plot_identity_line:
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()

            min_limit = min(x_limits[0], y_limits[0])
            max_limit = max(x_limits[1], y_limits[1])

            ax.plot(
                [min_limit, max_limit],
                [min_limit, max_limit],
                c="black",
                linestyle="dashed",
                linewidth=1,
            )


class GridPlot2D(ScatterPlot2D):
    """
    A class for creating grid plots using scatter points and connecting lines, extending `ScatterPlot2D`.

    Inherits functionality from `ScatterPlot2D` and customizes it for displaying structured grids with connecting lines.

    ### Plot Components
    - Scatter points are arranged in a grid.
    - Points in the same row or column are connected with lines.
    - Each point can be color-coded based on its position.
    """

    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = (5, 5),
        **kwargs,
    ) -> None:
        """
        - `figsize` (Tuple[float, float], optional): Figure size. Defaults to `(5, 5)`.
        - `**kwargs`: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            figsize=figsize,
            **kwargs,
        )

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        grid: Tensor,
        height: int,
        width: int,
        line_width: int = 2,
        point_size: int = 200,
        *args,
        **kwargs,
    ) -> None:
        """
        Plots a 2D grid with scatter points and connecting lines.

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `grid` (Tensor): Tensor of shape `(height*width, 2)` representing the grid points.
        - `height` (int): Number of rows in the grid.
        - `width` (int): Number of columns in the grid.
        - `line_width` (int, optional): Line width for the grid connections. Defaults to `2`.
        - `point_size` (int, optional): Size of the scatter points. Defaults to `200`.

        ### Plot Components
        - Points in the grid are connected by lines along rows and columns.
        - Each point is displayed as a scatter point, color-coded by its position.
        """
        assert (
            len(grid.shape) == 2 and grid.size(dim=-1) == 2
        ), f"grid shape must be (height*width, 2), got {tuple(grid.shape)}"

        # Plot horizontal lines (connect points in the same row)
        for i in range(height):
            row_points = grid[i * width : (i + 1) * width]
            ax.plot(
                row_points[:, 0],
                row_points[:, 1],
                color="black",
                linewidth=line_width,
                zorder=1,
            )

        # Plot vertical lines (connect points in the same column)
        for j in range(width):
            col_points = grid[j::width]
            ax.plot(
                col_points[:, 0],
                col_points[:, 1],
                color="black",
                linewidth=line_width,
                zorder=1,
            )

        # Plot points
        ax.scatter(
            grid[:, 0],
            grid[:, 1],
            s=point_size,
            c=torch.linspace(0, 1, len(grid)),
            vmin=0,
            vmax=1,
            zorder=2,
            *args,
            **kwargs,
        )

        ax.axis("off")
