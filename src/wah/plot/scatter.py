import torch

from ..typing import Axes, Figure, Iterable, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "GridPlot2D",
    "ScatterPlot2D",
]


class ScatterPlot2D(Plot2D):
    """
    A class for creating 2D scatter plots with customizable settings.

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
    - `clabel (Optional[str])`: The label for the colorbar.
    - `clim (Optional[Tuple[float, float]])`: The limits for the colorbar.
    - `cticks (Optional[Tuple[float, float]])`: The tick values for the colorbar.
    - `cticklabels (Optional[Tuple[float, float]])`: The tick labels for the colorbar.
    - `grid_alpha (Optional[float])`: The alpha transparency for the grid.

    ### Methods
    - `__init__(...)`: Initializes the ScatterPlot2D object with customizable settings.
    - `_plot(fig, ax, x, y, c, *args, **kwargs) -> None`: Plots the scatter plot with optional color mapping.
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
        - `clabel (Optional[str])`: The label for the colorbar. Defaults to None.
        - `clim (Optional[Tuple[float, float]])`: The limits for the colorbar. Defaults to None.
        - `cticks (Optional[Tuple[float, float]])`: The tick values for the colorbar. Defaults to None.
        - `cticklabels (Optional[Tuple[float, float]])`: The tick labels for the colorbar. Defaults to None.
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
        Plots the scatter plot with optional color mapping.

        ### Parameters
        - `fig (Figure)`: The Matplotlib figure object.
        - `ax (Axes)`: The Matplotlib axes object.
        - `x (Iterable[float])`: The x-coordinates of the points.
        - `y (Iterable[float])`: The y-coordinates of the points.
        - `c (Optional[Iterable[float]])`: The color values for the points. Defaults to None.
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Returns
        - `None`

        ### Notes
        - This method generates a scatter plot of the data using Matplotlib. If `c` is provided, a colorbar is added to the plot.
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
    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = (5, 5),
    ) -> None:
        super().__init__(
            figsize=figsize,
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
