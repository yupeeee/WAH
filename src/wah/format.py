import subprocess

from . import path as _path
from .misc.typing import List, Namespace, Path

__all__ = [
    "main",
]


def load_py_fpaths(
    root: Path,
) -> List[Path]:
    return [
        fpath
        for fpath in _path.walk(root, absolute=True)
        if _path.splitext(fpath) == ".py"
    ]


def format_files(
    fpaths: List[Path],
) -> None:
    subprocess.run(["isort"] + fpaths)
    subprocess.run(["black"] + fpaths)


def main(args: Namespace):
    fpaths = []
    for root in args.roots:
        fpaths.extend(load_py_fpaths(root))
    format_files(fpaths)
