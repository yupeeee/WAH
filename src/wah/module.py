from ast import literal_eval

from .typing import Any, Dict, List, Module, Optional, Tuple

__all__ = [
    "_getattr",
    "get_attrs",
    "get_module_name",
    "get_module_params",
    "get_named_modules",
    "get_valid_attr",
]


def _getattr(
    module: Module,
    attr: str,
) -> Module:
    """
    Retrieves a nested attribute from a module.

    ### Parameters
    - `module` (Module): The module from which to retrieve the attribute.
    - `attr` (str): The attribute string, which can include nested attributes separated by dots.

    ### Returns
    - `Module`: The module or submodule referred to by the attribute.

    ### Raises
    - `AttributeError`: If the attribute is not found.
    """
    try:
        for a in attr.split("."):
            if attr.isnumeric():
                module = module[int(a)]
            else:
                module = getattr(module, a)
        return module
    except AttributeError:
        raise


def get_attrs(
    model: Module,
    skip_dropout: Optional[bool] = True,
    skip_identity: Optional[bool] = True,
    specify: Optional[str] = None,
) -> List[str]:
    """
    Retrieves a list of valid attribute names from a model, with options to exclude dropout and identity layers,
    and to specify a particular type of module.

    ### Parameters
    - `model` (Module): The model from which to retrieve attribute names.
    - `skip_dropout` (bool, optional): Whether to skip layers that include dropout modules. Defaults to `True`.
    - `skip_identity` (bool, optional): Whether to skip layers that are instances of `torch.nn.Identity`. Defaults to `True`.
    - `specify` (str, optional): If provided, only returns attributes corresponding to this specific module type (e.g., "Conv2d").
    If `None`, all valid attributes are returned. Defaults to `None`.

    ### Returns
    - `List[str]`: A list of valid attribute names from the model that match the specified conditions.
    """
    names = get_named_modules(model)

    attrs: List[str] = []

    for name in names:
        attr = get_valid_attr(model, name)

        ############
        # Skip if: #
        ############

        # attr is not valid
        if attr is None:
            continue

        # module is dropout
        if skip_dropout and "drop" in attr:
            continue

        # module is Identity()
        if skip_identity and get_module_name(_getattr(model, attr)) == "Identity":
            continue

        if attr not in attrs:
            attrs.append(attr)

    valid_attrs = []

    for i in range(len(attrs) - 1):
        # if attrs[i] not in attrs[i + 1]: << cannot detect cases like ..., "norm", "fc_norm", ... (excludes "norm")
        if attrs[i + 1].replace(attrs[i], "")[0] != ".":
            valid_attrs.append(attrs[i])

    valid_attrs.append(attrs[-1])

    if specify is not None:
        valid_attrs = [
            attr
            for attr in valid_attrs
            if get_module_name(_getattr(model, attr)) == specify
        ]

    return valid_attrs


def get_module_name(
    module: Module,
) -> str:
    """
    Returns the class name of a module.

    ### Parameters
    - `module` (Module): The module whose class name is being retrieved.

    ### Returns
    - `str`: The name of the module's class.
    """
    return module.__class__.__name__


def get_module_params(
    module: Module,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Parses and returns the parameters of a module.

    ### Parameters
    - `module` (Module): The module whose parameters are being retrieved.

    ### Returns
    - `Tuple[List[Any], Dict[str, Any]]`: A tuple containing:
        - A list of positional arguments (args).
        - A dictionary of keyword arguments (kwargs).

    ### Notes
    - Uses string manipulation to extract arguments and keyword arguments from the module's string representation.
    """
    # "Module(param0, arg1=(param1, ), arg2=param2, ...)"
    s = str(module).replace("\n", "")

    # "param0, arg1=(param1, ), arg2=param2, ..."
    params = ")".join("(".join(s.split("(")[1:]).split(")")[:-1])

    # "param0,arg1=(param1,),arg2=param2,..."
    params = params.replace(" ", "")

    # ["param0", "arg1=(param1", ")", "arg2=param2", ...]
    params = params.split(",")

    # ["param0", "arg1=(param1,)", "arg2=param2", ]
    def process_params(params: List[str]):
        i = 0

        # adjusted to prevent index out of range error
        while i < len(params) - 1:
            if "(" in params[i] and ")" in params[i + 1]:
                yield f"{params[i]}{params[i + 1]}"

                # increment by 2 to skip the next element
                # since it's already paired
                i += 2
            else:
                yield params[i]
                i += 1

        # handle the case where the last element might not have been yielded
        if i < len(params):
            yield params[i]

    # args = [param0, ]
    # kwargs = {"arg1": (param1), "arg2": param2, ...}
    args = []
    kwargs = {}

    for param in process_params(params):
        if "=" not in param:
            args.append(literal_eval(param))
        else:
            k, v = param.split("=")
            kwargs[k] = literal_eval(v)

    return args, kwargs


def get_named_modules(
    model: Module,
) -> List[str]:
    """
    Retrieves the names of all named submodules within a model.

    ### Parameters
    - `model` (Module): The model from which to retrieve submodule names.

    ### Returns
    - `List[str]`: A list of submodule names.
    """
    return [name for name, _ in model.named_modules()]


def get_valid_attr(
    model: Module,
    attr: str,
) -> str:
    """
    Retrieves a valid attribute string from a model.

    ### Parameters
    - `model` (Module): The model from which to retrieve the attribute.
    - `attr` (str): The attribute string, potentially containing nested attributes.

    ### Returns
    - `str`: A valid attribute string, or `None` if no valid attribute is found.
    """
    attrs = attr.split(".")
    valid_attr = []

    try:
        module = _getattr(model, attrs[0])
        valid_attr.append(attrs[0])

        for a in attrs[1:]:
            module = _getattr(module, a)
            valid_attr.append(a)
    except AttributeError:
        pass

    return ".".join(valid_attr) if len(valid_attr) else None
