import matplotlib.pyplot as plt

from .. import path as _path
from ..typing import Axes, Path, Tensor

__all__ = [
    "ImShow",
]


class ImShow:
    """
    A class for displaying images using matplotlib, supporting grayscale and RGB images.

    ### Plot Components
    - Images are displayed in a grid of subplots with customizable dimensions.
    - Supports both grayscale and RGB image formats.
    - Optionally removes axis lines and labels from the subplots.
    """

    def __init__(
        self,
        height: int,
        width: int,
        scale: float = 1.0,
        no_axis: bool = True,
    ) -> None:
        """
        - `height` (int): Number of rows in the grid of subplots.
        - `width` (int): Number of columns in the grid of subplots.
        - `scale` (float, optional): Scaling factor for the size of each subplot. Defaults to `1.0`.
        - `no_axis` (bool, optional): If `True`, hides the axis lines and labels. Defaults to `True`.
        """
        self.height = height
        self.width = width
        self.scale = scale
        self.no_axis = no_axis

    def _is_grayscale(
        self,
        image: Tensor,
    ) -> bool:
        """
        Checks whether the image is grayscale.

        ### Parameters
        - `image` (Tensor): Image tensor to check.

        ### Returns
        - `bool`: `True` if the image is grayscale, `False` otherwise.
        """
        return image.dim() == 2 or (image.dim() == 3 and image.size(0) == 1)

    def plot(
        self,
        images: Tensor,
    ) -> None:
        """
        Plots a grid of images using matplotlib.

        ### Parameters
        - `images` (Tensor): A batch of images to display.
        Should have shape `(N, C, H, W)` where `N` is the number of images, `C` is the number of channels, and `H`, `W` are the height and width of each image.

        ### Plot Components
        - Images are arranged in a grid of size `(height, width)` with customizable scaling.
        - Grayscale images are displayed using a grayscale colormap.
        - RGB images are displayed with their original colors.
        - Axis labels can be hidden based on the `no_axis` attribute.
        """
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
        Displays the plotted images.
        """
        plt.show()

    def save(
        self,
        save_path: Path,
        use_auto_settings: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves the plotted images to the specified file path.

        ### Parameters
        - `save_path` (Path): The path where the images will be saved.
        - `use_auto_settings` (bool, optional): Whether to apply default saving settings (DPI, tight layout). Defaults to `True`.
        - `**kwargs`: Additional arguments for `plt.savefig`.
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
