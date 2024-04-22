from ast import literal_eval

from ..typing import (
    Any,
    Dict,
    List,
    Module,
    Tuple,
)

__all__ = [
    "_getattr",
    "get_valid_attr",
    "get_module_name",
    "get_module_params",
    "get_named_modules",
    "get_attrs",
]


def _getattr(
    module: Module,
    attr: str,
) -> Module:
    """
    Returns the value of the attribute (attr) from the module.
    If the specified attribute does not exist within the module,
    an AttributeError exception is raised.
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


def get_valid_attr(
    model: Module,
    attr: str,
) -> str:
    """
    Returns the longest valid nested attribute (attr) name
    that exists in the model.
    If the specified attribute does not exist within the model,
    returns None.
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


def get_module_name(
    module: Module,
) -> str:
    """
    Returns the class name of the module.
    """
    return module.__class__.__name__


def get_module_params(
    module: Module,
) -> Tuple[List[Any], Dict[str, Any]]:
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
    return [name for name, _ in model.named_modules()]


def get_attrs(
    model: Module,
) -> List[str]:
    """
    Returns all valid module names (attrs) of a given neural network model.
    """
    names = get_named_modules(model)

    attrs = []

    for name in names:
        attr = get_valid_attr(model, name)

        # Skip if attr is not valid
        if attr is None:
            continue

        if attr not in attrs:
            attrs.append(attr)

    return attrs
