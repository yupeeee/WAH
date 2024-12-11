import torch
from torch import nn

from ...module import _getattr, get_attrs

# from ...tensor import flatten_batch
from ...typing import Dict, List, Module, RemovableHandle, Tensor, Union

# from torchvision.models.feature_extraction import (
#     create_feature_extractor,
# )  # get_graph_node_names,


__all__ = [
    "FeatureExtractor",
]


class FeatureExtractor(nn.Module):
    """
    A class to extract features from specified layers of a given model.

    This class registers hooks on the specified layers (attributes) of a model and extracts their intermediate outputs
    during the forward pass. It can return the inputs or outputs of the layers based on the settings.

    ### Attributes
    - `model` (Module): The model from which features are to be extracted.
    - `attrs` (List[str]): A list of layer attribute names to extract features from. If `None`, all valid attributes are used.
    - `penultimate_only` (bool): If `True`, only returns features from the second-to-last layer. Defaults to `False`.
    - `return_inputs` (bool): If `True`, returns the inputs to the specified layers instead of the outputs. Defaults to `False`.

    ### Methods
    - `forward(x: Tensor) -> Union[Dict[str, Tensor], Tensor]`: Performs a forward pass through the model and extracts
       features from the specified layers.
    - `train() -> None`: Sets the model to training mode.
    - `eval() -> None`: Sets the model to evaluation mode.
    """

    def __init__(
        self,
        model: Module,
        attrs: List[str] = None,
        penultimate_only: bool = False,
        return_inputs: bool = False,
    ) -> None:
        """
        - `model` (Module): The model from which features are to be extracted.
        - `attrs` (List[str], optional): A list of attribute names representing the layers to extract features from.
          If `None`, valid attributes are automatically retrieved using `get_attrs()`. Defaults to `None`.
        - `penultimate_only` (bool, optional): If `True`, only the second-to-last layer's features are returned.
          Defaults to `False`.
        - `return_inputs` (bool, optional): If `True`, the inputs to the specified layers are returned instead of the outputs.
          Defaults to `False`.
        """
        super().__init__()

        self.model = model
        self.attrs = attrs if attrs is not None else get_attrs(model)
        self.penultimate_only = penultimate_only
        self.return_inputs = return_inputs

    def forward(self, x: Tensor) -> Union[Dict[str, Tensor], Tensor]:
        hooks: List[RemovableHandle] = []
        features: List[Tensor] = []

        def hook_fn(module, input, output):
            if not self.return_inputs:
                features.append(output)
            else:
                features.append(input)

        for attr in self.attrs:
            hook_handle: RemovableHandle = _getattr(
                self.model, attr
            ).register_forward_hook(hook_fn)
            hooks.append(hook_handle)

        with torch.no_grad():
            _ = self.model(x)

        for hook_handle in hooks:
            hook_handle.remove()

        features: Dict[str, Tensor] = dict(
            (self.attrs[i], features[i]) for i in range(len(self.attrs))
        )

        if self.penultimate_only:
            features = {self.attrs[-2]: features[self.attrs[-2]]}

        if len(features.keys()) == 1:
            features = list(features.values())[0]

        return features


# class FeatureExtractor(Module):
#     """
#     A class for extracting features from intermediate layers of a model.

#     This class leverages `torchvision.models.feature_extraction.create_feature_extractor`
#     to retrieve features from specific layers of a model. It supports extracting features
#     from all layers or only the penultimate layer.

#     ### Attributes
#     - `model` (Module): The model from which features are extracted.
#     - `penultimate_only` (bool): If `True`, only the penultimate layer's features are returned. Defaults to `False`.
#     - `feature_layers` (Dict[str, str]): A dictionary mapping original layer names to internal feature layer names.
#     - `feature_extractor` (Module): The feature extractor created from the model and the specified layers.
#     - `checked_layers` (bool): Whether the layers have been verified through a forward pass.

#     ### Methods
#     - `forward(x: Tensor) -> Dict[str, Tensor]`: Extracts features from the specified layers in the model.
#     - `_check_layers(x: Tensor) -> None`: Validates and updates the list of layers based on a forward pass.
#     """

#     def __init__(
#         self,
#         model: Module,
#         penultimate_only: bool = False,
#     ) -> None:
#         """
#         - `model` (Module): The model from which to extract features.
#         - `penultimate_only` (bool, optional): If `True`, only the penultimate layer's features are returned. Defaults to `False`.
#         """
#         super().__init__()

#         self.model = model
#         self.penultimate_only = penultimate_only

#         # _, layers = get_graph_node_names(model)
#         layers = get_attrs(model)

#         self.feature_layers = dict(
#             (layer, f"{i}_{layer}") for i, layer in enumerate(layers)
#         )

#         self.feature_extractor = create_feature_extractor(
#             model=model,
#             return_nodes=self.feature_layers,
#         )

#         self.checked_layers: bool = False

#     def forward(self, x):
#         """
#         Extracts features from the specified layers in the model.

#         ### Parameters
#         - `x` (Tensor): The input tensor to pass through the model.

#         ### Returns
#         - `Dict[str, Tensor]`: A dictionary of extracted features from the specified layers.
#         """
#         if not self.checked_layers:
#             self._check_layers(x)

#         features: Dict[str, Tensor] = self.feature_extractor(x)

#         for layer, feature in features.items():
#             if isinstance(feature, tuple):
#                 feature = [f for f in feature if f is not None]
#                 feature = torch.cat(feature, dim=0)

#                 features[layer] = feature

#         if self.penultimate_only:
#             features = features[list(features.keys())[-2]]

#         return features

#     def _check_layers(self, x) -> None:
#         """
#         Validates and updates the list of layers by performing a forward pass.

#         This method checks which layers produce valid outputs and updates the feature extractor to only include these layers.

#         ### Parameters
#         - `x` (Tensor): The input tensor to test the layers of the model.
#         """
#         assert not self.checked_layers

#         layers = []

#         with torch.no_grad():
#             features = self.feature_extractor(x)

#             for layer, i_layer in self.feature_layers.items():
#                 # skip input
#                 if layer == "x":
#                     continue

#                 # skip Identity()
#                 if isinstance(_getattr(self.model, layer), torch.nn.Identity):
#                     continue

#                 # # skip if output type has no attr len
#                 # # TypeError: object of type 'int' has no len()
#                 # try:
#                 #     _ = len(features[i_layer])
#                 # except TypeError:
#                 #     continue

#                 # # skip weight/bias/gamma/etc
#                 # if len(features[i_layer]) != len(x):
#                 #     continue

#                 try:
#                     _ = flatten_batch(features[i_layer])
#                     torch.cuda.empty_cache()

#                     layers.append(layer)

#                 except BaseException:
#                     continue

#         self.feature_layers = dict(
#             (layer, f"{i}_{layer}") for i, layer in enumerate(layers)
#         )

#         self.feature_extractor = create_feature_extractor(
#             model=self.model,
#             return_nodes=self.feature_layers,
#         )

#         # delattr(self, "model")
#         self.checked_layers = True
