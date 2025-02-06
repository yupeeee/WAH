import argparse

from .train import DATASET_CONFIGS
from .train import main as train


def main():
    parser = argparse.ArgumentParser(
        prog="wah", description="Wah command-line interface."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # train
    train_parser = subparsers.add_parser("train", help="Train a model on a dataset")
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Name of the dataset. Available options: {', '.join(DATASET_CONFIGS.keys())}",
    )
    train_parser.add_argument(
        "--dataset-root",
        type=str,
        required=False,
        default=".",
        help="Root directory where datasets are/will be stored",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model. Use timm.list_models() to see available options",
    )
    train_parser.add_argument(
        "--replace",
        type=str,
        required=False,
        default=None,
        help="Replace layers of one type with another. Format: from1:to1,from2:to2,... (e.g. 'bn:ln')",
    )
    train_parser.add_argument(
        "--cfg-path",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    train_parser.add_argument(
        "--log-root",
        type=str,
        required=False,
        default="./logs",
        help="Root directory where training logs will be stored",
    )
    train_parser.add_argument(
        "--version",
        type=str,
        required=False,
        default=None,
        help="Name of the version to identify this training run",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the last checkpoint",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="Device to run the model on",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
