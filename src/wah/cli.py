import argparse
import subprocess

__all__ = [
    "main",
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="wah")
    subparsers = parser.add_subparsers(dest="command", required=True)

    format_parser = subparsers.add_parser(
        "format",
        help="Format a directory of files or a file.",
    )
    format_parser.add_argument(
        "path",
        required=False,
        default=".",
        help="The directory or file to format.",
    )

    args = parser.parse_args()

    if args.command == "format":
        subprocess.run(["isort", args.path], check=True)
        subprocess.run(["black", args.path], check=True)
