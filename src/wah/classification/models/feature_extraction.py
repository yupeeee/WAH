import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)  # get_graph_node_names,

from ...tensor import flatten_batch
from ...typing import Dict, Module, Tensor
from ...utils.module import _getattr, get_attrs

__all__ = [
    "FeatureExtractor",
]


class FeatureExtractor(Module):
    """
    A class for extracting features from intermediate layers of a model.

    This class leverages `torchvision.models.feature_extraction.create_feature_extractor`
    to retrieve features from specific layers of a model. It supports extracting features
    from all layers or only the penultimate layer.

    ### Attributes
    - `model` (Module): The model from which features are extracted.
    - `penultimate_only` (bool): If `True`, only the penultimate layer's features are returned. Defaults to `False`.
    - `feature_layers` (Dict[str, str]): A dictionary mapping original layer names to internal feature layer names.
    - `feature_extractor` (Module): The feature extractor created from the model and the specified layers.
    - `checked_layers` (bool): Whether the layers have been verified through a forward pass.

    ### Methods
    - `forward(x: Tensor) -> Dict[str, Tensor]`: Extracts features from the specified layers in the model.
    - `_check_layers(x: Tensor) -> None`: Validates and updates the list of layers based on a forward pass.
    """

    def __init__(
        self,
        model: Module,
        penultimate_only: bool = False,
    ) -> None:
        """
        - `model` (Module): The model from which to extract features.
        - `penultimate_only` (bool, optional): If `True`, only the penultimate layer's features are returned. Defaults to `False`.
        """
        super().__init__()

        self.model = model
        self.penultimate_only = penultimate_only

        # _, layers = get_graph_node_names(model)
        layers = get_attrs(model)

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
        Extracts features from the specified layers in the model.

        ### Parameters
        - `x` (Tensor): The input tensor to pass through the model.

        ### Returns
        - `Dict[str, Tensor]`: A dictionary of extracted features from the specified layers.
        """
        if not self.checked_layers:
            self._check_layers(x)

        features: Dict[str, Tensor] = self.feature_extractor(x)

        for layer, feature in features.items():
            if isinstance(feature, tuple):
                feature = [f for f in feature if f is not None]
                feature = torch.cat(feature, dim=0)

                features[layer] = feature

        if self.penultimate_only:
            features = features[list(features.keys())[-2]]

        return features

    def _check_layers(self, x) -> None:
        """
        Validates and updates the list of layers by performing a forward pass.

        This method checks which layers produce valid outputs and updates the feature extractor to only include these layers.

        ### Parameters
        - `x` (Tensor): The input tensor to test the layers of the model.
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

        # delattr(self, "model")
        self.checked_layers = True
