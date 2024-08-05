import numpy as np

from ..typing import (
    Axes,
    Dict,
    Figure,
    Iterable,
    List,
    Optional,
    Tuple,
)
from .base import Plot2D

__all__ = [
    "DistPlot2D",
]


def _dict_to_mat(
    data_dict: Dict[float, List[float]],
) -> Tuple[List[float], np.ndarray]:
    """
    Converts a dictionary of data to a matrix format suitable for plotting.

    ### Parameters
    - `data_dict (Dict[float, List[float]])`: A dictionary where keys are floats and values are lists of floats.

    ### Returns
    - `Tuple[List[float], np.ndarray]]`: A tuple containing a list of keys and a 2D numpy array of values.

    ### Notes
    - The function ensures the values are converted to a 2D numpy array.
    - It asserts that the resulting array has 2 dimensions.
    """
    keys = []
    vals = []

    for k, v in data_dict.items():
        keys.append(float(k))
        vals.append(v)

    vals = np.array(vals)
    assert len(vals.shape) == 2

    return keys, vals


class DistPlot2D(Plot2D):
    """
    A class for creating 2D distribution plots (mean, min, max, and quartile values) with customizable settings.

    ### Methods
    - `plot`: Creates the plot with the specified settings.
    - `show`: Displays the plot.
    - `save`: Saves the plot to the specified path with optional settings.
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
        data_dict: Dict[float, List[float]],
        *args,
        **kwargs,
    ) -> None:
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

        # q1, q2, q3
        ax.plot(
            x,
            q2s,
            linewidth=1,
            color="black",
            zorder=3,
        )
        ax.fill_between(
            x,
            q1s,
            q3s,
            alpha=0.30,
            color="black",
            edgecolor=None,
            zorder=2,
        )
