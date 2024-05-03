import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from ..typing import (
    Module,
    Tensor,
)

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

        _, layers = get_graph_node_names(model)

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

        self.checked_layers = False

    def forward(self, x):
        if not self.checked_layers:
            self.check_layers(x)

        features = self.feature_extractor(x)

        for layer, feature in features.items():
            if isinstance(feature, tuple):
                feature = [f for f in feature if f is not None]
                feature = torch.cat(feature, dim=0)

                features[layer] = feature

        return features

    def check_layers(self, x) -> None:
        assert not self.checked_layers

        layers = []

        with torch.no_grad():
            features = self.feature_extractor(x)

            for layer, i_layer in self.feature_layers.items():
                if layer == "x":
                    continue

                try:
                    _ = self.flatten_feature(features[i_layer], x)
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

        self.checked_layers = True

    @staticmethod
    def flatten_feature(feature, x) -> Tensor:
        # vit: self_attention
        if isinstance(feature, tuple):
            feature = [f for f in feature if f is not None]
            feature = torch.cat(feature, dim=0)

        feature = feature.reshape(len(x), -1)

        return feature
