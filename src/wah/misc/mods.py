from ast import literal_eval

from .typing import Any, Dict, List, Module, Optional, Tuple

__all__ = [
    "getargs",
    "getattrs",
    "getmod",
    "getname",
    "setmod",
]


def clean_attr(
    module: Module,
    attr: str,
) -> str:
    """Clean module attributes.

    ### Args
        - `module` (Module): Module to clean attributes of
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
    module: Module,
    specify: Optional[str] = None,
    max_depth: Optional[int] = None,
    skip_dropout: Optional[bool] = True,
    skip_identity: Optional[bool] = True,
) -> List[str]:
    """Get all module attributes.

    ### Args
        - `module` (Module): Module to get attributes of
        - `specify` (Optional[str]): Only return modules of this type
        - `max_depth` (Optional[int]): Maximum depth of attributes to return
        - `skip_dropout` (Optional[bool]): Skip dropout layers
        - `skip_identity` (Optional[bool]): Skip identity layers

    ### Returns
        - `List[str]`: List of module attributes

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
    >>> getattrs(model)
    ['conv1', 'relu1', 'layer1.conv2', 'layer1.relu2', 'layer1.layer2.conv3']
    >>> getattrs(model, specify='Conv2d')
    ['conv1', 'layer1.conv2', 'layer1.layer2.conv3']
    >>> getattrs(model, max_depth=1)
    ['conv1', 'relu1', 'layer1']
    >>> getattrs(model, max_depth=2)
    ['conv1', 'relu1', 'layer1.conv2', 'layer1.relu2', 'layer1.layer2']
    ```
    """
    if max_depth is not None:
        assert (
            isinstance(max_depth, int) and max_depth > 0
        ), "max_depth must be a positive integer"

    # Get all module names
    attrs = []
    for name, _ in module.named_modules():
        attr = clean_attr(module, name)

        # Skip invalid attributes and filtered modules
        if (
            attr is None
            or (skip_dropout and "drop" in attr)
            or (skip_identity and getname(getmod(module, attr)) == "Identity")
            or (max_depth is not None and attr.count(".") >= max_depth)
        ):
            continue

        attrs.append(attr)

    # Remove duplicate attributes
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


def getargs(
    module: Module,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Get the arguments of a module.

    ### Args
        - `module` (Module): Module to get arguments of

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


def getmod(
    module: Module,
    attr: str,
) -> Module:
    """Get a module attribute by string path.

    ### Args
        - `module` (Module): Base module to search from
        - `attr` (str): String path to attribute (e.g. "conv1.0.weight")

    ### Returns
        - `Module`: Module at the specified path

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


def getname(
    module: Module,
) -> str:
    """Get the name of a module.

    ### Args
        - `module` (Module): Module to get name of

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


def setmod(
    module: Module,
    attr: str,
    new_module: Module,
) -> None:
    """Set a module attribute by string path.

    ### Args
        - `module` (Module): Base module to modify
        - `attr` (str): String path to attribute (e.g. "conv1.0.weight")
        - `new_module` (Module): New module to set at the path

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
