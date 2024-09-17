import matplotlib.pyplot as plt

from .. import path as _path
from ..typing import Axes, Path, Tensor

__all__ = [
    "ImShow",
]


class ImShow:
    def __init__(
        self,
        height: int,
        width: int,
        scale: float = 1.0,
        no_axis: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.scale = scale
        self.no_axis = no_axis

    def _is_grayscale(
        self,
        image: Tensor,
    ) -> bool:
        """Helper function to determine if an image is grayscale."""
        return image.dim() == 2 or (image.dim() == 3 and image.size(0) == 1)

    def plot(
        self,
        images: Tensor,
    ) -> None:
        num_images = images.size(0)
        fig, axes = plt.subplots(
            nrows=self.height,
            ncols=self.width,
            figsize=(self.width * self.scale, self.height * self.scale),
        )
        axes = axes.flatten()

        for i in range(min(num_images, len(axes))):
            ax: Axes = axes[i]
            img = images[i]

            if self._is_grayscale(img):
                img = img.squeeze(0)  # Remove channel dimension for grayscale images
                ax.imshow(img.cpu(), cmap="gray")
            else:
                img = img.permute(1, 2, 0).cpu()  # Convert from (C, H, W) to (H, W, C)
                ax.imshow(img)

            if self.no_axis:
                ax.axis("off")

        # Turn off any remaining axes if there are fewer images than subplots
        for j in range(num_images, len(axes)):
            ax: Axes = axes[j]
            ax.axis("off")

        plt.tight_layout()

    def show(self) -> None:
        """
        Displays the plot.

        ### Returns:
        - `None`

        ### Notes:
        - This method calls `plt.show()` to display the plot in an interactive window.
        """
        plt.show()

    def save(
        self,
        save_path: Path,
        use_auto_settings: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves the plot to the specified path with optional settings.

        ### Parameters:
        - `save_path (Path)`: The path to save the plot.
        - `use_auto_settings (bool)`: Whether to use automatic settings for saving the plot. Defaults to True.
        - `**kwargs`: Additional keyword arguments for saving the plot.

        ### Notes:
        - This method saves the plot to the specified `save_path` using Matplotlib's `plt.savefig()` function.
        - If `use_auto_settings` is True, the method automatically adjusts settings such as DPI and padding to optimize the saved plot.
        - The directory specified in `save_path` is created if it does not exist.
        """
        if use_auto_settings:
            kwargs["dpi"] = 300
            kwargs["bbox_inches"] = "tight"
            kwargs["pad_inches"] = 0.01

        save_path = _path.clean(save_path)
        save_dir, _ = _path.split(save_path)
        _path.mkdir(save_dir)

        plt.savefig(save_path, **kwargs)
        plt.draw()
        plt.close("all")
