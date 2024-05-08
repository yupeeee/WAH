import subprocess
from typing import (
    Literal,
    Tuple,
)

__all__ = [
    "run",
]


def run(
    args: str,
    terminal: Literal[
        "cmd",
        "powershell",
    ] = "cmd",
) -> Tuple[bool, str]:
    if terminal == "cmd":
        args = "" + args
    elif terminal == "powershell":
        args = "powershell -Command " + args
    else:
        raise ValueError(
            f"terminal must be one of ['cmd', 'powershell', ], got {terminal}"
        )

    result = subprocess.run(args, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        return True, result.stdout

    else:
        return False, result.stderr
