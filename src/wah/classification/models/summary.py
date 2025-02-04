import torch

from ...misc.mods import getattrs, getmod, getname
from ...misc.typing import Dict, Module, Optional, Sequence, Tuple, Union

__all__ = [
    "summary",
]


def summary(
    model: Module,
    input_shape: Sequence[int],
    input_dtype: type = torch.float32,
    eval: Optional[bool] = True,
    skip_dropout: Optional[bool] = True,
    skip_identity: Optional[bool] = True,
    print_summary: Optional[bool] = True,
) -> Dict[str, Dict[str, Union[str, Tuple[int, ...]]]]:
    """Get a summary of a PyTorch model's layers and shapes.

    ### Args
        - `model` (Module): PyTorch model to analyze
        - `input_shape` (Sequence[int]): Shape of input tensor
        - `input_dtype` (type): Data type of input tensor. Defaults to `torch.float32`.
        - `eval` (bool, optional): Whether to set model to eval mode. Defaults to `True`.
        - `skip_dropout` (bool, optional): Whether to skip dropout layers. Defaults to `True`.
        - `skip_identity` (bool, optional): Whether to skip identity layers. Defaults to `True`.
        - `print_summary` (bool, optional): Whether to print summary. Defaults to `True`.

    ### Returns
        - `Dict[str, Dict[str, Union[str, Tuple[int, ...]]]]`: Dictionary containing layer information

    ### Example
    ```python
    >>> model = torch.nn.Sequential(OrderedDict([
    ...     ('conv', torch.nn.Conv2d(3, 64, 3)),
    ...     ('relu', torch.nn.ReLU()),
    ...     ('pool', torch.nn.MaxPool2d(2))
    ... ]))
    >>> summary = summary(model, input_shape=(1, 3, 32, 32))
        Layer       Input -> Output
    -------------------------------------
    (0) conv        (1, 3, 32, 32)
        (Conv2d)    -> (1, 64, 30, 30)
    -------------------------------------
    (1) relu        (1, 64, 30, 30)
        (ReLU)      -> (1, 64, 30, 30)
    -------------------------------------
    (2) pool        (1, 64, 30, 30)
        (MaxPool2d) -> (1, 64, 15, 15)
    -------------------------------------
    #params: 1792
    >>> print(summary)
    {
        'conv': {
            'module_name': 'Conv2d',
            'input_shape': (1, 3, 32, 32),
            'output_shape': (1, 64, 30, 30)
        },
        'relu': {
            'module_name': 'ReLU',
            'input_shape': (1, 64, 30, 30),
            'output_shape': (1, 64, 30, 30)
        },
        'pool': {
            'module_name': 'MaxPool2d',
            'input_shape': (1, 64, 30, 30),
            'output_shape': (1, 64, 15, 15)
        }
    }
    ```
    """
    # Set model to eval mode if requested
    if eval:
        model.eval()

    # Get list of layer attribute paths to analyze
    layers = getattrs(
        module=model,
        skip_dropout=skip_dropout,
        skip_identity=skip_identity,
    )

    # Initialize storage for layer info
    summaries: Dict[str, Dict[str, Union[str, Tuple[int, ...]]]] = {}

    # Setup forward hook to capture shapes
    def hook_fn(layer_path: str):
        def _hook(module, inputs, outputs):
            # Get input shape
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            input_shape = tuple(inputs.shape)

            # Get output shape
            output_shape = tuple(outputs.shape)

            # Store layer info
            summaries[layer_path] = {
                "module_name": getname(module),
                "input_shape": input_shape,
                "output_shape": output_shape,
            }

        return _hook

    # Register hooks for each layer
    hooks = []
    for layer in layers:
        module = getmod(model, layer)
        hook = module.register_forward_hook(hook_fn(layer))
        hooks.append(hook)

    # Run forward pass
    device = (
        next(model.parameters()).device
        if not hasattr(model, "device")
        else model.device
    )
    with torch.no_grad():
        model(torch.randn(input_shape, dtype=input_dtype, device=device))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print summary if requested
    if print_summary:
        # Calculate column widths
        idx_width = len(str(len(layers) - 1)) + 3
        layer_width = max(len(info["module_name"]) for info in summaries.values()) + 3
        shape_width = (
            max(
                max(
                    len(str(info["input_shape"])),
                    len("-> " + str(info["output_shape"])),
                )
                for info in summaries.values()
            )
            + 3
        )

        # Print header
        header = f"{'Layer':<{layer_width}}{'Input -> Output':<{shape_width}}"
        print(f"{'':<{idx_width}}{header}")
        print("-" * (idx_width + layer_width + shape_width))

        # Print each layer
        for i, layer_path in enumerate(layers):
            info = summaries[layer_path]
            _module_name = info["module_name"]
            _input_shape = info["input_shape"]
            _output_shape = info["output_shape"]
            print(f"{f'({i})':<{idx_width}}{layer_path:<{layer_width}}{_input_shape}")
            print(
                f"{'':<{idx_width}}{f'({_module_name})':<{layer_width}}-> {_output_shape}"
            )
            print("-" * (idx_width + layer_width + shape_width))

        # Print parameter count
        print(f"#params: {sum(p.numel() for p in model.parameters())}")

    return summaries
