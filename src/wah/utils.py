__all__ = [
    "clean",
]


def clean(s):
    s = "_".join(filter(
        lambda c: str.isidentifier(c) or str.isdecimal(c), s
    ))

    return s
