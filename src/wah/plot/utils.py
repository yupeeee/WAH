import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..typing import (
    Axes,
    AxIndex,
    Color,
    Colormap,
    Colornorm,
    Figure,
    NDArray,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    # ax
    "_get_ax",
    # colorbar
    "_init_colors",
    "_show_colorbar",
]


def _get_ax(
    axes: Union[Axes, NDArray],
    index: AxIndex,
) -> Axes:
    if isinstance(axes, Axes):
        assert index == 0, f"There is only a single Axes!"
        return axes

    if isinstance(index, int):
        return axes[index]
    else:
        ax = axes
        for i in index:
            ax = ax[i]
        return ax


def _init_colors(
    c: Union[str, float, Sequence[float]],
    cmap: str,
    cmin: float,
    cmax: float,
) -> Tuple[Union[Color, Sequence[Color]], Colormap, Colornorm]:
    if isinstance(c, str):
        return c, None, None

    else:
        cmin = min(c) if cmin is None else cmin
        cmax = max(c) if cmax is None else cmax

        colormap = cm.get_cmap(cmap)
        colornorm = mcolors.Normalize(vmin=cmin, vmax=cmax)
        c = colormap(colornorm(c))

        return c, colormap, colornorm


def _show_colorbar(
    fig: Figure,
    ax: Axes,
    colormap: Colormap,
    colornorm: Colornorm,
    label: Optional[str] = None,
    ticks: Optional[Sequence[float]] = None,
    ticklabels: Optional[Sequence[str]] = None,
) -> None:
    colorbar = fig.colorbar(cm.ScalarMappable(norm=colornorm, cmap=colormap), ax=ax)
    colorbar.set_label(label)
    if ticks is not None:
        colorbar.set_ticks(ticks=ticks, labels=ticklabels)
