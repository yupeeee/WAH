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
    def __init__(
        self,
        model: Module,
        penultimate_only: bool = False,
    ) -> None:
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
