import torch

from ...module import _getattr, get_attrs, get_module_name
from ...typing import (
    Dict,
    Module,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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
    # Set model to eval mode if requested
    if eval:
        model.eval()

    # Get list of layer attribute paths to analyze
    layers = get_attrs(
        model=model,
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
                "module_name": get_module_name(module),
                "input_shape": input_shape,
                "output_shape": output_shape,
            }

        return _hook

    # Register hooks for each layer
    hooks = []
    for layer_path in layers:
        module = _getattr(model, layer_path)
        hook = module.register_forward_hook(hook_fn(layer_path))
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
        layer_width = max(len(path) for path in layers) + 3
        shape_strs = []
        for info in summaries.values():
            shape_strs.append(str(info["input_shape"]))
            shape_strs.append("-> " + str(info["output_shape"]))
        shape_width = max(len(s) for s in shape_strs) + 3

        # Print header
        header = f"{'Layer':<{layer_width}}{'Input -> Output':<{shape_width}}"
        print(f"{'':<{idx_width}}{header}")
        print("-" * (idx_width + layer_width + shape_width))

        # Print each layer
        for i, layer_path in enumerate(layers):
            info = summaries[layer_path]
            module_name = info["module_name"]
            input_shape = info["input_shape"]
            output_shape = info["output_shape"]
            print(f"{f'({i})':<{idx_width}}{layer_path:<{layer_width}}{input_shape}")
            print(
                f"{'':<{idx_width}}{f'({module_name})':<{layer_width}}-> {output_shape}"
            )
            print("-" * (idx_width + layer_width + shape_width))

        # Print parameter count
        num_params = sum(p.numel() for p in model.parameters())
        print(f"#params: {num_params}")

    return summaries
