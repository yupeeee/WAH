import torch

from ...module import _getattr, get_attrs, get_module_name
from ...typing import Dict, List, Module, Optional, RemovableHandle, Sequence, Tuple

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
) -> None:
    if eval:
        model.eval()

    attrs = get_attrs(
        model=model, skip_dropout=skip_dropout, skip_identity=skip_identity
    )

    hooks: List[RemovableHandle] = []
    model_summary = Dict[
        str,
        Tuple[
            Tuple[int, int, int, int],
            Tuple[int, int, int, int],
        ],
    ]

    def hook_fn(module, input, output):
        # module
        module_name = get_module_name(module)

        # input
        if isinstance(input, tuple):
            assert len(input) == 1
            input = input[0]
        input_shape = tuple(input.shape)

        # output
        output_shape = tuple(output.shape)

        model_summary[module_name] = [input_shape, output_shape]

    for attr in attrs:
        hook_handle: RemovableHandle = _getattr(model, attr).register_forward_hook(
            hook_fn
        )
        hooks.append(hook_handle)

    with torch.no_grad():
        _ = model(torch.randn(size=input_shape, dtype=input_dtype))

    for hook_handle in hooks:
        hook_handle.remove()

    module_names = list(model_summary.keys())
    input_shapes = [model_summary[module_name][0] for module_name in module_names]
    output_shapes = [model_summary[module_name][1] for module_name in module_names]

    # width (+3: +2 for brackets, +1 for spacing)
    idx_width = len(str(len(attrs) - 1)) + 3
    layer_width = max(len(attr) for attr in attrs) + 3
    input_width = max(len(str(input_shapes[i])) for i in range(len(attrs))) + 3
    output_width = (
        max(len("-> " + str(output_shapes[i])) for i in range(len(attrs))) + 3
    )
    input_output_width = max(input_width, output_width)

    # print header
    header = f"{'Layer':<{layer_width}}{'Input -> Output':<{input_output_width}}"
    print(f"{'':<{idx_width}}{header}")
    print("-" * (idx_width + layer_width + input_output_width))

    # print summary
    for i, (attr, module, input_shape, output_shape) in enumerate(
        zip(attrs, module_names, input_shapes, output_shapes)
    ):
        print(f"{f'({i})':<{idx_width}}{attr:<{layer_width}}{input_shape}")
        print(f"{' ' * idx_width}{f'({module})':<{layer_width}}-> {output_shape}")
        print("-" * (idx_width + layer_width + input_output_width))

    # print #params
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#params: {num_params}")

    return model_summary
