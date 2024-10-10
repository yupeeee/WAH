import numpy as np

from ..typing import Axes, Dict, Figure, Iterable, List, Optional, Tensor, Tuple, Union
from .base import Plot2D

__all__ = [
    "DistPlot2D",
]


def _dict_to_mat(
    data_dict: Dict[float, List[float]],
) -> Tuple[List[float], np.ndarray]:
    """
    Converts a dictionary of data into a tuple of keys and a 2D numpy array.

    ### Parameters
    - `data_dict` (Dict[float, List[float]]): Dictionary where keys are floats and values are lists of floats.

    ### Returns
    - `Tuple[List[float], np.ndarray]`: A tuple with keys and the values converted into a 2D numpy array.
    """
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
    A class for creating 2D distribution plots using matplotlib, extending the `Plot2D` class.

    Inherits plot settings and customization from `Plot2D`, and adds functionality to plot distributions with means,
    quantiles, and ranges.

    ### Plot Components
    - Means are plotted as red dots.
    - Minimum and maximum values are shown as shaded areas.
    - Quartiles are shown as a shaded area between Q1 and Q3, and Q2 is plotted as a line.
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
        data: Union[Tensor, Dict[float, List[float]]],
        *args,
        **kwargs,
    ) -> None:
        """
        Plots distribution data from a dictionary, showing the means, quantiles, and ranges.

        ### Parameters
        - `fig` (Figure): Matplotlib figure object.
        - `ax` (Axes): Matplotlib axes object.
        - `data` (Union[Tensor, Dict[float, List[float]]]): Data to plot, either as a tensor or a dictionary.

        ### Plot Components
        - Means are plotted as red dots.
        - Minimum and maximum values are shown as shaded areas.
        - Quartiles are shown as a shaded area between Q1 and Q3, and Q2 is plotted as a line.
        """
        if isinstance(data, Tensor):
            assert len(data.shape) == 2, f"Input tensor must be 2D, got {data.shape}"

            x = np.arange(len(data))
            y = np.array(data)

        elif isinstance(data, dict):
            x, y = _dict_to_mat(data)

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

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
