from ast import literal_eval
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "getmod",
    "setmod",
    "getname",
    "getargs",
    "getattrs",
    "summary",
]


def getmod(
    module: torch.nn.Module,
    attr: str,
) -> torch.nn.Module:
    """Get a module attribute by string path.

    ### Args
        - `module` (torch.nn.Module): Base module to search from
        - `attr` (str): String path to attribute (e.g. "conv1.0.weight")

    ### Returns
        - `torch.nn.Module`: Module at the specified path

    ### Raises
        - `AttributeError`: If attribute path is not found

    ### Example
    ```python
    >>> from collections import OrderedDict
    >>> model = torch.nn.Sequential(OrderedDict([
    ...     ('conv1', torch.nn.Sequential(
    ...         torch.nn.Conv2d(1, 20, 5),
    ...         torch.nn.BatchNorm2d(20)
    ...     )),
    ...     ('relu', torch.nn.ReLU()),
    ...     ('conv2', torch.nn.Conv2d(20, 64, 5))
    ... ]))
    >>> conv1 = getmod(model, "conv1")           # Get first Sequential block
    >>> conv1_conv = getmod(model, "conv1.0")    # Get Conv2d from first block
    >>> conv1_bn = getmod(model, "conv1.1")      # Get BatchNorm from first block
    >>> relu = getmod(model, "relu")             # Get ReLU layer
    >>> conv2 = getmod(model, "conv2")           # Get second Conv2d layer
    >>> weight = getmod(model, "conv1.0.weight") # Get weights of first Conv2d
    ```
    """
    try:
        for a in attr.split("."):
            if a.isnumeric():
                module = module[int(a)]
            else:
                module = getattr(module, a)
        return module
    except AttributeError:
        raise AttributeError(f"Could not find attribute '{attr}' in module")


def setmod(
    module: torch.nn.Module,
    attr: str,
    new_module: torch.nn.Module,
) -> None:
    """Set a module attribute by string path.

    ### Args
        - `module` (torch.nn.Module): Base module to modify
        - `attr` (str): String path to attribute (e.g. "conv1.0.weight")
        - `new_module` (torch.nn.Module): New module to set at the path

    ### Returns
        - `None`

    ### Raises
        - `AttributeError`: If attribute path is not found

    ### Example
    ```python
    >>> from collections import OrderedDict
    >>> model = torch.nn.Sequential(OrderedDict([
    ...     ('conv1', torch.nn.Sequential(
    ...         torch.nn.Conv2d(1, 20, 5),
    ...         torch.nn.BatchNorm2d(20)
    ...     )),
    ...     ('relu', torch.nn.ReLU()),
    ...     ('conv2', torch.nn.Conv2d(20, 64, 5))
    ... ]))
    >>> # Replace the first Conv2d with a new one
    >>> setmod(model, "conv1.0", torch.nn.Conv2d(1, 32, 3))
    >>> # Replace the ReLU with a LeakyReLU
    >>> setmod(model, "relu", torch.nn.LeakyReLU())
    >>> # Update weights of first Conv2d
    >>> setmod(model, "conv1.0.weight", torch.randn(32, 1, 3, 3))
    ```
    """
    try:
        *attrs, final_attr = attr.split(".")
        for a in attrs:
            if a.isnumeric():
                module = module[int(a)]
            else:
                module = getattr(module, a)
        setattr(module, final_attr, new_module)
    except AttributeError:
        raise AttributeError(f"Could not find attribute '{attr}' in module")


def getname(
    module: torch.nn.Module,
) -> str:
    """Get the name of a module.

    ### Args
        - `module` (torch.nn.Module): Module to get name of

    ### Returns
        - `str`: Name of module class

    ### Example
    ```python
    >>> import torch
    >>> getname(torch.nn.Conv2d(1, 20, 5))
    'Conv2d'
    >>> getname(torch.nn.Sequential())
    'Sequential'
    ```
    """
    return module.__class__.__name__


def getargs(
    module: torch.nn.Module,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Get the arguments of a module.

    ### Args
        - `module` (torch.nn.Module): Module to get arguments of

    ### Returns
        - `Tuple[List[Any], Dict[str, Any]]`: Tuple containing:
            - List of positional arguments
            - Dict of keyword arguments

    ### Example
    ```python
    >>> import torch
    >>> args, kwargs = getargs(torch.nn.Conv2d(1, 20, 5))
    >>> args
    [1, 20]
    >>> kwargs
    {'kernel_size': (5, 5), 'stride': (1, 1)}
    >>> args, kwargs = getargs(torch.nn.Linear(in_features=10, out_features=5))
    >>> args
    []
    >>> kwargs
    {'in_features': 10, 'out_features': 5, 'bias': True}
    ```
    """
    # Get the module's signature from its __repr__
    sig_str = str(module)

    # Extract everything between first ( and last )
    params_str = sig_str[sig_str.find("(") + 1 : sig_str.rfind(")")]

    args = []
    kwargs = {}

    # Track nested parentheses for proper parsing
    paren_count = 0
    current_param = []

    for char in params_str:
        if char == "(":
            paren_count += 1
            current_param.append(char)
        elif char == ")":
            paren_count -= 1
            current_param.append(char)
        elif char == "," and paren_count == 0:
            # Process completed parameter
            param = "".join(current_param).strip()
            if param:
                if "=" in param:
                    k, v = param.split("=", 1)
                    kwargs[k.strip()] = literal_eval(v.strip())
                else:
                    args.append(literal_eval(param))
            current_param = []
        else:
            current_param.append(char)

    # Process final parameter
    param = "".join(current_param).strip()
    if param:
        if "=" in param:
            k, v = param.split("=", 1)
            kwargs[k.strip()] = literal_eval(v.strip())
        else:
            args.append(literal_eval(param))

    return args, kwargs


def clean_attr(
    module: torch.nn.Module,
    attr: str,
) -> str:
    """Clean module attributes.

    ### Args
        - `module` (torch.nn.Module): Module to clean attributes of
        - `attr` (str): Attribute string to clean

    ### Returns
        - `str`: Cleaned attribute string, or None if no valid attributes found

    ### Example
    ```python
    >>> import torch
    >>> conv = torch.nn.Conv2d(3, 64, 3)
    >>> clean_attr(conv, 'weight.shape')
    'weight'
    >>> clean_attr(conv, 'invalid.attr')
    None
    ```
    """
    attrs = attr.split(".")
    valid_attrs = []
    curr_module = module
    for attr_part in attrs:
        try:
            curr_module = getmod(curr_module, attr_part)
            valid_attrs.append(attr_part)
        except AttributeError:
            break
    return ".".join(valid_attrs) if valid_attrs else None


def getattrs(
    module: torch.nn.Module,
    specify: Optional[str] = None,
    max_depth: Optional[int] = None,
    skip_dropout: Optional[bool] = True,
    skip_identity: Optional[bool] = True,
) -> List[str]:
    """List all submodule attribute paths.

    ### Args
        - `module` (torch.nn.Module): Module to list submodules of
        - `specify` (Optional[str]): Only return modules of this type
        - `max_depth` (Optional[int]): Maximum depth of attributes to return
        - `skip_dropout` (Optional[bool]): Skip dropout layers
        - `skip_identity` (Optional[bool]): Skip identity layers

    ### Returns
        - `List[str]`: List of submodule attribute paths

    ### Example
    ```python
    >>> from collections import OrderedDict
    >>> import torch
    >>> model = torch.nn.Sequential(OrderedDict([
    ...     ('conv1', torch.nn.Conv2d(3, 64, 3)),
    ...     ('relu1', torch.nn.ReLU()),
    ...     ('layer1', torch.nn.Sequential(OrderedDict([
    ...         ('conv2', torch.nn.Conv2d(64, 128, 3)),
    ...         ('relu2', torch.nn.ReLU()),
    ...         ('layer2', torch.nn.Sequential(OrderedDict([
    ...             ('conv3', torch.nn.Conv2d(128, 256, 3))
    ...         ])))
    ...     ])))
    ... ]))
    >>> listmods(model)
    ['conv1', 'relu1', 'layer1.conv2', 'layer1.relu2', 'layer1.layer2.conv3']
    >>> listmods(model, specify='Conv2d')
    ['conv1', 'layer1.conv2', 'layer1.layer2.conv3']
    >>> listmods(model, max_depth=1)
    ['conv1', 'relu1', 'layer1']
    >>> listmods(model, max_depth=2)
    ['conv1', 'relu1', 'layer1.conv2', 'layer1.relu2', 'layer1.layer2']
    ```
    """
    if max_depth is not None:
        assert (
            isinstance(max_depth, int) and max_depth > 0
        ), "max_depth must be a positive integer"

    # Get all submodule attribute paths
    attrs = []
    for name, _ in module.named_modules():
        attr = clean_attr(module, name)

        # Skip invalid attribute paths and filtered submodules
        if (
            attr is None
            or (skip_dropout and "drop" in attr)
            or (skip_identity and getname(getmod(module, attr)) == "Identity")
            or (max_depth is not None and attr.count(".") >= max_depth)
        ):
            continue

        attrs.append(attr)

    # Remove duplicate attribute paths
    attrs = list(dict.fromkeys(attrs))

    # Filter out parent modules that are part of child module names
    valid_attrs = []
    for i, attr in enumerate(attrs[:-1]):
        # Check if current attr is not a parent path of next attr
        if not attrs[i + 1].startswith(attr + "."):
            valid_attrs.append(attr)
    valid_attrs.append(attrs[-1])

    # Filter by specified module type if needed
    if specify is not None:
        valid_attrs = [
            attr for attr in valid_attrs if getname(getmod(module, attr)) == specify
        ]

    return valid_attrs


def summary(
    model: torch.nn.Module,
    input_shape: Sequence[int],
    input_dtype: type = torch.float32,
    eval: Optional[bool] = True,
    skip_dropout: Optional[bool] = True,
    skip_identity: Optional[bool] = True,
    print_summary: Optional[bool] = True,
) -> Dict[str, Dict[str, Union[str, Tuple[int, ...]]]]:
    """Get a summary of a PyTorch model's layers and shapes.

    ### Args
        - `model` (torch.nn.Module): PyTorch model to analyze.
        - `input_shape` (Sequence[int]): Shape of input tensor.
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
