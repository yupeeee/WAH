from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict, List, Set

from ..misc.typing import Any

__all__ = [
    "config",
]


def config(cls):
    # Convert mutable defaults to default_factory
    for name, value in cls.__annotations__.items():
        if hasattr(cls, name):
            default = getattr(cls, name)
            if isinstance(default, (List, Dict, Set)):
                # Replace mutable default with default_factory
                setattr(cls, name, field(default_factory=lambda x=default: type(x)()))

    cls = dataclass(cls)

    # Post-init hook to safely init default_factory values
    original_post_init = getattr(cls, "__post_init__", None)

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            # If None but default_factory exists, create a new instance
            if value is None and f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
        if original_post_init:
            original_post_init(self)

    cls.__post_init__ = __post_init__

    # Add .dict()
    def dict(self):
        return asdict(self)

    cls.dict = dict

    # Add .from_dict()
    @classmethod
    def from_dict(cls_, d: Dict[str, Any]):
        return cls_(**d)

    # Add .copy()
    def copy(self):
        return self.from_dict(self.dict())

    # Add .update()
    def update(self, **kwargs):
        # Get all field names from the dataclass
        field_names = [f.name for f in fields(self.__class__)]

        valid_kwargs = self.dict()

        # Update with new values, filtering to only include valid field names
        valid_kwargs.update({k: v for k, v in kwargs.items() if k in field_names})

        # Update self with new values
        for k, v in valid_kwargs.items():
            setattr(self, k, v)

    cls.from_dict = from_dict
    cls.copy = copy
    cls.update = update

    return cls
