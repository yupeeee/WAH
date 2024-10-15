import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA

from ... import path, utils
from ...plot.base import Plot2D
from ...typing import (
    Any,
    Axes,
    Config,
    Dataset,
    Devices,
    Dict,
    Figure,
    List,
    Module,
    NDArray,
    Optional,
    Path,
    Tensor,
    Tuple,
)
from ..models.load import load_state_dict
from ..test.loss import LossTest

__all__ = [
    "proj_train_path_to_2d",
    "TrainPathPlot2D",
]


def load_weights(
    model: Module,
    weights_dir: Path,
    epoch_interval: Optional[int] = 1,
) -> NDArray:
    weights: List[Tensor] = []

    # locate weights to load
    state_dict_paths = [
        f
        for f in path.ls(
            path=weights_dir,
            fext=".ckpt",
            sort=True,
            absolute=True,
        )
        if "last.ckpt" not in f
    ][::epoch_interval]

    # load weights
    for state_dict_path in tqdm.tqdm(
        state_dict_paths,
        desc=f"Loading {len(state_dict_paths)} weights from {weights_dir}",
    ):
        load_state_dict(model, state_dict_path)
        weights_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        weights.append(weights_vec.unsqueeze(dim=0))

    # weights.shape: (num_epochs // epoch_interval, #params)
    weights: Tensor = torch.cat(weights, dim=0)

    return weights.numpy()


def perform_pca(
    weights: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, float]:
    print(f"Projecting weights using PCA... ({weights.shape[-1]}D -> 2D)")

    pca = PCA(n_components=2)

    weights_proj = pca.fit_transform(weights)
    weights_mean = pca.mean_
    basis = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_.sum()

    print(f"done. (Explained variance ratio: {explained_variance_ratio * 100:.2f}%)")

    return weights_proj, weights_mean, basis, explained_variance_ratio


def create_2d_grid(
    weights_proj: NDArray,
    num_steps: Optional[int] = 10,
) -> Tuple[NDArray, NDArray]:
    # Create a grid of points around the projected weights for contour plotting
    x_min, x_max = weights_proj[:, 0].min(), weights_proj[:, 0].max()
    y_min, y_max = weights_proj[:, 1].min(), weights_proj[:, 1].max()

    # Padding (10%)
    x_eps = (x_max - x_min) * 0.1
    y_eps = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_eps, x_max + x_eps
    y_min, y_max = y_min - y_eps, y_max + y_eps

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, num_steps), np.linspace(y_min, y_max, num_steps)
    )

    return x_grid, y_grid


def load_2d_coordinates_to_model_weights(
    model: Module,
    x: float,
    y: float,
    weights_mean: NDArray,
    basis: NDArray,
) -> None:
    weights_vec = x * basis[0] + y * basis[1] + weights_mean
    torch.nn.utils.vector_to_parameters(
        vec=torch.from_numpy(weights_vec),
        parameters=model.parameters(),
    )


def proj_train_path_to_2d(
    dataset: Dataset,
    model: Module,
    train_dir: Path,
    epoch_interval: Optional[int] = 1,
    num_steps: Optional[int] = 10,
    devices: Optional[Devices] = "auto",
) -> Dict[str, Any]:
    # load config
    config: Config = utils.load_yaml_to_dict(path.join(train_dir, "hparams.yaml"))

    # load weights
    weights_dir = path.join(train_dir, "checkpoints")
    weights = load_weights(model, weights_dir, epoch_interval)

    # PCA projection
    weights_proj, weights_mean, basis, explained_variance_ratio = perform_pca(
        weights,
    )

    # create 2d grid of weights for loss contour
    loss_xs, loss_ys = create_2d_grid(weights_proj, num_steps)
    loss_zs = np.zeros_like(loss_xs)

    # Loop over grid points and compute loss by converting 2D coordinates back to full-dimensional weights
    for k in tqdm.trange(
        num_steps * num_steps,
        desc=f"Generating loss contour...",
    ):
        i, j = divmod(k, num_steps)

        # Get the 2D point and transform it back to the original weight space
        load_2d_coordinates_to_model_weights(
            model=model,
            x=loss_xs[i, j],
            y=loss_ys[i, j],
            weights_mean=weights_mean,
            basis=basis,
        )

        # Compute the loss for the current set of weights
        test = LossTest(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            mixup_alpha=config["mixup_alpha"],
            cutmix_alpha=config["cutmix_alpha"],
            label_smoothing=config["criterion_cfg"]["label_smoothing"],
            seed=config["seed"],
            devices=devices,
            verbose=False,
        )
        loss_zs[i, j] = test(model, dataset)

    print("done.")

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "basis": basis,
        "weights_proj": weights_proj,
        "loss_xs": loss_xs,
        "loss_ys": loss_ys,
        "loss_zs": loss_zs,
    }


class TrainPathPlot2D(Plot2D):
    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        title: Optional[str] = None,
        cmap: Optional[str] = "viridis",
    ) -> None:
        super().__init__(
            figsize,
            fontsize,
            title,
            xlabel=None,
            xlim=None,
            xticks=None,
            xticklabels=None,
            ylabel=None,
            ylim=None,
            yticks=None,
            yticklabels=None,
            grid_alpha=0.0,
        )
        self.cmap = cmap

        self.train_path_data: Dict[str, Any] = None

    def make_data(
        self,
        dataset: Dataset,
        model: Module,
        train_dir: Path,
        epoch_interval: Optional[int] = 1,
        num_steps: Optional[int] = 10,
        seed: Optional[int] = 0,
        devices: Optional[Devices] = "auto",
    ) -> None:
        utils.seed(seed)

        self.train_path_data = proj_train_path_to_2d(
            dataset=dataset,
            model=model,
            train_dir=train_dir,
            epoch_interval=epoch_interval,
            num_steps=num_steps,
            devices=devices,
        )

    def load_data(self, path: Path) -> None:
        self.train_path_data = torch.load(path)

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        assert (
            self.train_path_data is not None
        ), f"No data available for plotting. Please generate new data or load existing data."

        # Title: explained variance ratio
        self.title = f"Explained variance ratio: {self.train_path_data['explained_variance_ratio'] * 100:.2f}%"

        # Loss contour
        contour = ax.contourf(
            self.train_path_data["loss_xs"],
            self.train_path_data["loss_ys"],
            self.train_path_data["loss_zs"],
            levels=100,
            cmap=self.cmap,
            alpha=0.8,
        )
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Loss")

        # Train trajectory
        xs = self.train_path_data["weights_proj"][:, 0]
        ys = self.train_path_data["weights_proj"][:, 1]

        ax.quiver(
            xs[:-1],
            ys[:-1],
            np.diff(xs),
            np.diff(ys),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
        )
