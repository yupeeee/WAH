import subprocess
from ..typing import (
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
    """
    Runs a command in the specified terminal and captures the output.

    ### Parameters
    - `args (str)`: The command to run in the terminal.
    - `terminal (Literal["cmd", "powershell"])`: The terminal to use for running the command. Defaults to "cmd".

    ### Returns
    - `Tuple[bool, str]`: A tuple where the first element is a boolean indicating success, and the second element is the command output or error message.

    ### Raises
    - `ValueError`: If the specified terminal is not one of ['cmd', 'powershell'].

    ### Notes
    - This function uses `subprocess.run` to execute the command in the specified terminal.
    - The output is captured and returned as a string.
    - If the command executes successfully, the function returns `True` and the command's standard output.
    - If the command fails, the function returns `False` and the error message.
    """
    if terminal == "cmd":
        args = "" + args
    elif terminal == "powershell":
        args = "powershell -Command " + args
    else:
        raise ValueError(
            f"terminal must be one of ['cmd', 'powershell'], got {terminal}"
        )

    result = subprocess.run(args, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        return True, result.stdout
    else:
        return False, result.stderr
