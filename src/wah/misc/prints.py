from typing import Literal

__all__ = [
    "stylish",
]


def stylish(
    text: str,
    style: Literal[
        "bold",
        "dim",
        "italic",
        "underline",
        "blink_slow",
        "blink_fast",
        "highlight",
        "hidden",
        "strikethrough",
    ] = None,
    color: Literal[
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
    ] = None,
) -> str:
    # ANSI code mappings for styles and colors
    style_codes = {
        "bold": "1",
        "dim": "2",
        "italic": "3",
        "underline": "4",
        "blink_slow": "5",
        "blink_fast": "6",
        "highlight": "7",
        "hidden": "8",
        "strikethrough": "9",
        None: "",
    }
    color_codes = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
        None: "",
    }

    style_code = style_codes.get(style, None)
    color_code = color_codes.get(color, None)

    seq = ""
    if style_code is not None and style_code != "":
        seq += f"\033[{style_code}m"
    if color_code is not None and color_code != "":
        seq += f"\033[{color_code}m"

    end_seq = "\033[0m" if seq else ""
    text = f"{seq}{text}{end_seq}"

    return text
