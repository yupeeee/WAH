import torch

from ...module import _getattr, get_attrs, get_module_name
from ...typing import (
    Dict,
    List,
    Module,
    Optional,
    RemovableHandle,
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
    if eval:
        model.eval()

    layers = get_attrs(
        model=model,
        skip_dropout=skip_dropout,
        skip_identity=skip_identity,
    )

    hooks: List[RemovableHandle] = []
    module_names: List[str] = []
    input_shapes: List[Tuple[int, ...]] = []
    output_shapes: List[Tuple[int, ...]] = []

    def hook_fn(module, input, output):
        # module
        module_name = get_module_name(module)
        module_names.append(module_name)

        # input
        if isinstance(input, tuple):
            assert len(input) == 1
            input = input[0]
        input_shape = tuple(input.shape)
        input_shapes.append(input_shape)

        # output
        output_shape = tuple(output.shape)
        output_shapes.append(output_shape)

    for layer in layers:
        hook_handle: RemovableHandle = _getattr(model, layer).register_forward_hook(
            hook_fn
        )
        hooks.append(hook_handle)

    device = (
        model.device if hasattr(model, "device") else next(model.parameters()).device
    )
    with torch.no_grad():
        _ = model(torch.randn(size=input_shape, dtype=input_dtype, device=device))

    for hook_handle in hooks:
        hook_handle.remove()

    model_summary: Dict[str, Dict[str, Union[str, Tuple[int, ...]]]] = {
        layer: {
            "module_name": module_name,
            "input_shape": input_shape,
            "output_shape": output_shape,
        }
        for layer, module_name, input_shape, output_shape in zip(
            layers, module_names, input_shapes, output_shapes
        )
    }

    if print_summary:
        # width (+3: +2 for brackets, +1 for spacing)
        idx_width = len(str(len(module_names) - 1)) + 3
        layer_width = max(len(name) for name in module_names) + 3
        input_width = max(len(str(shape)) for shape in input_shapes) + 3
        output_width = max(len("-> " + str(shape)) for shape in output_shapes) + 3
        input_output_width = max(input_width, output_width)

        # print header
        header = f"{'Layer':<{layer_width}}{'Input -> Output':<{input_output_width}}"
        print(f"{'':<{idx_width}}{header}")
        print("-" * (idx_width + layer_width + input_output_width))

        # print summary
        for i, name in enumerate(module_names):
            info = model_summary[name]
            print(f"{f'({i})':<{idx_width}}{name:<{layer_width}}{info['input_shape']}")
            print(
                f"{' ' * idx_width}{f'({name})':<{layer_width}}-> {info['output_shape']}"
            )
            print("-" * (idx_width + layer_width + input_output_width))

        # print #params
        num_params = sum(p.numel() for p in model.parameters())
        print(f"#params: {num_params}")

    return model_summary
