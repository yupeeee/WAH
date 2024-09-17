import numpy as np

from ..typing import Axes, Dict, Figure, Iterable, List, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "DistPlot2D",
]


def _dict_to_mat(
    data_dict: Dict[float, List[float]],
) -> Tuple[List[float], np.ndarray]:
    keys = []
    vals = []

    for k, v in data_dict.items():
        k = float(k)
        if not isinstance(v, list):
            v = [float(x) for x in v]

        keys.append(k)
        vals.append(v)

    vals = np.array(vals)
    assert len(vals.shape) == 2

    return keys, vals


class DistPlot2D(Plot2D):
    """
    A class for creating 2D distribution plots (mean, min, max, and quartile values) with customizable settings.

    ### Attributes
    - `figsize` (Optional[Tuple[float, float]])`: The size of the figure.
    - `fontsize` (Optional[float])`: The font size for the plot text.
    - `title` (Optional[str])`: The title of the plot.
    - `xlabel` (Optional[str])`: The label for the x-axis.
    - `xlim` (Optional[Tuple[float, float]])`: The limits for the x-axis.
    - `xticks` (Optional[Iterable[float]])`: The tick values for the x-axis.
    - `xticklabels` (Optional[Iterable[str]])`: The tick labels for the x-axis.
    - `ylabel` (Optional[str])`: The label for the y-axis.
    - `ylim` (Optional[Tuple[float, float]])`: The limits for the y-axis.
    - `yticks` (Optional[Iterable[float]])`: The tick values for the y-axis.
    - `yticklabels` (Optional[Iterable[str]])`: The tick labels for the y-axis.
    - `grid_alpha` (Optional[float])`: The alpha transparency for the grid.

    ### Methods
    - `__init__(...)`: Initializes the DistPlot2D object with customizable settings.
    - `_plot(fig, ax, data_dict, *args, **kwargs) -> None`: Plots the 2D distribution plot based on the provided data dictionary.
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
        data_dict: Dict[float, List[float]],
        *args,
        **kwargs,
    ) -> None:
        """
        Plots the 2D distribution plot based on the provided data dictionary.

        ### Parameters
        - `fig (Figure)`: The Matplotlib figure object.
        - `ax (Axes)`: The Matplotlib axes object.
        - `data_dict (Dict[float, List[float]])`: A dictionary where keys are x-values and values are lists of y-values representing distributions.
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Returns
        - `None`

        ### Notes
        - This method generates a scatter plot of the means, fills the area between min and max values, and adds lines and areas for quartiles (Q1, Q2, Q3).
        """
        x, y = _dict_to_mat(data_dict)

        means = np.mean(y, axis=-1)
        maxs = np.max(y, axis=-1)
        mins = np.min(y, axis=-1)
        q1s = np.quantile(y, 0.25, axis=-1)
        q2s = np.quantile(y, 0.50, axis=-1)
        q3s = np.quantile(y, 0.75, axis=-1)

        # means
        ax.scatter(
            x,
            means,
            marker="o",
            s=2,
            color="red",
            zorder=4,
        )

        # mins, maxs
        ax.fill_between(
            x,
            mins,
            maxs,
            alpha=0.15,
            color="black",
            edgecolor=None,
            zorder=1,
        )

        # q2
        ax.plot(
            x,
            q2s,
            linewidth=1,
            color="black",
            zorder=3,
        )
        # q1, q3
        ax.fill_between(
            x,
            q1s,
            q3s,
            alpha=0.30,
            color="black",
            edgecolor=None,
            zorder=2,
        )
