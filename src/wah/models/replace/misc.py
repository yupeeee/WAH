import torch

from ...typing import Any, Dict, List, Module, Optional, Tuple, Union
from ..modules import get_module_name, get_module_params

__all__ = [
    "replace_keys_in_kwargs",
    "replace_module",
    "ReplaceModule",
]


def replace_keys_in_kwargs(
    kwargs: Dict[str, Any],
    keymaps: List[Tuple[str, str]],
) -> Dict[str, Any]:
    for key_i, key_f in keymaps:
        if None in [key_i, key_f] and key_i is not None:
            kwargs.pop(key_i)
            continue

        if key_i in kwargs.keys():
            kwargs[key_f] = kwargs.pop(key_i)

    return kwargs


def replace_module(
    model: Module,
    attr: str,
    replace_with: Module,
    test_replacement: Optional[torch.Tensor] = None,
) -> None:
    attrs = attr.split(".")

    module = model

    for a in attrs[:-1]:
        if a.isnumeric():
            module = module[int(a)]

        else:
            module = getattr(module, a)

    setattr(module, attrs[-1], replace_with)

    if test_replacement is not None:
        try:
            _ = model(test_replacement)
            return

        except BaseException:
            raise

    else:
        return


class ReplaceModule:
    target_module_name: Union[str, List[str]]
    keymaps: List[Tuple[str, str]]
    _replacement_module: Module

    def __init__(self, ) -> None:
        if isinstance(self.target_module_name, str):
            self.target_module_name = [self.target_module_name]

    def __call__(
        self,
        module: Module,
        use_cuda: bool = False,
    ) -> Any:
        raise NotImplementedError

    def replacement_module(
        self,
        module: Module,
        use_cuda: bool = False,
    ) -> Module:
        assert get_module_name(module) in self.target_module_name

        args, kwargs = get_module_params(module)

        if self._replacement_module is None:
            return module, args, kwargs

        else:
            if self.keymaps is not None:
                kwargs = replace_keys_in_kwargs(kwargs, self.keymaps)

            if use_cuda is not None:
                kwargs["device"] = "cuda" if use_cuda else "cpu"

            return self._replacement_module(*args, **kwargs)
