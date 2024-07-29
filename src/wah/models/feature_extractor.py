import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    # get_graph_node_names,
)

from ..typing import (
    Module,
)
from ..utils.tensor import flatten_batch
from .modules import _getattr, get_attrs

__all__ = [
    "FeatureExtractor",
]


class FeatureExtractor(Module):
    """
    A feature extractor class for extracting features from a specified model.

    ### Parameters
    - `model (Module)`: The PyTorch model from which to extract features.
    - `penultimate_only (bool, optional)`: If `True`, only extracts features from the penultimate layer. Defaults to `False`.

    ### Methods
    - `forward`: Performs a forward pass through the feature extractor and returns the extracted features.

    ### Notes
    - This class uses the `torchvision.models.feature_extraction.create_feature_extractor` to create a feature extractor for the given model.
    - If `penultimate_only` is `True`, only the penultimate layer's features are extracted.
    - If `penultimate_only` is `False`, features from all layers are extracted.
    """

    def __init__(
        self,
        model: Module,
        penultimate_only: bool = False,
    ) -> None:
        """
        - `model (Module)`: The PyTorch model from which to extract features.
        - `penultimate_only (bool, optional)`: If `True`, only extracts features from the penultimate layer. Defaults to `False`.
        """
        super().__init__()

        self.model = model

        # _, layers = get_graph_node_names(model)
        layers = get_attrs(model)

        if penultimate_only:
            self.feature_layers = {
                layers[-2]: "features",
            }
        else:
            self.feature_layers = dict(
                (layer, f"{i}_{layer}") for i, layer in enumerate(layers)
            )

        self.feature_extractor = create_feature_extractor(
            model=model,
            return_nodes=self.feature_layers,
        )

        self.checked_layers: bool = False

    def forward(self, x):
        """
        Performs a forward pass through the feature extractor and returns the extracted features.

        ### Parameters
        - `x (Tensor)`: The input tensor.

        ### Returns
        - `Dict[str, Tensor]`: A dictionary of extracted features with layer names as keys.

        ### Notes
        - If the layers have not been checked yet, this method will call `_check_layers` first.
        - Handles cases where the feature output is a tuple by concatenating non-`None` elements.
        """
        if not self.checked_layers:
            self._check_layers(x)

        features = self.feature_extractor(x)

        for layer, feature in features.items():
            if isinstance(feature, tuple):
                feature = [f for f in feature if f is not None]
                feature = torch.cat(feature, dim=0)

                features[layer] = feature

        return features

    def _check_layers(self, x) -> None:
        """
        Checks and filters the layers for feature extraction.

        ### Parameters
        - `x (Tensor)`: The input tensor.

        ### Returns
        - `None`

        ### Notes
        - This method filters out layers that are not suitable for feature extraction.
        - Layers with outputs that do not match the input length or that cause errors during flattening are excluded.
        - The `feature_layers` attribute is updated to include only the valid layers.
        """
        assert not self.checked_layers

        layers = []

        with torch.no_grad():
            features = self.feature_extractor(x)

            for layer, i_layer in self.feature_layers.items():
                # skip input
                if layer == "x":
                    continue

                # skip Identity()
                if isinstance(_getattr(self.model, layer), torch.nn.Identity):
                    continue

                # # skip if output type has no attr len
                # # TypeError: object of type 'int' has no len()
                # try:
                #     _ = len(features[i_layer])
                # except TypeError:
                #     continue

                # # skip weight/bias/gamma/etc
                # if len(features[i_layer]) != len(x):
                #     continue

                try:
                    _ = flatten_batch(features[i_layer])
                    torch.cuda.empty_cache()

                    layers.append(layer)

                except BaseException:
                    continue

        self.feature_layers = dict(
            (layer, f"{i}_{layer}") for i, layer in enumerate(layers)
        )

        self.feature_extractor = create_feature_extractor(
            model=self.model,
            return_nodes=self.feature_layers,
        )

        delattr(self, "model")
        self.checked_layers = True
