from . import classification as lib
from .misc import cuda as _cuda
from .misc import dicts as _dicts
from .misc import path as _path
from .misc.typing import Config, Dataset, Module, Namespace, Trainer

__all__ = [
    "main",
]

DATASET_CONFIGS = {
    "cifar10": {
        "class_name": "CIFAR10",
        "num_classes": 10,
        "crop_size": 32,
        "resize_size": 32,
    },
    "cifar100": {
        "class_name": "CIFAR100",
        "num_classes": 100,
        "crop_size": 32,
        "resize_size": 32,
    },
    "imagenet": {
        "class_name": "ImageNet",
        "num_classes": 1000,
        "crop_size": 224,
        "resize_size": 256,
    },
    "stl10": {
        "class_name": "STL10",
        "num_classes": 10,
        "crop_size": 96,
        "resize_size": 96,
    },
}


def load_dataset(
    args: Namespace,
    config: Config,
    train: bool = False,
) -> Dataset:
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}. Currently supported datasets are: {DATASET_CONFIGS.keys()}"
        )
    # Adjust batch size for multi-GPU training
    num_devices = (
        len(config["devices"].split(":")[-1].split(","))
        if ":" in config["devices"]
        else 1
    )
    config["batch_size"] = int(config["batch_size"] / num_devices)
    # Get dataset config and class
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = getattr(lib.datasets, dataset_config["class_name"])
    # Update config with dataset parameters
    config.update(
        {
            "num_classes": dataset_config["num_classes"],
            "crop_size": dataset_config["crop_size"],
            "resize_size": dataset_config["resize_size"],
        }
    )
    # Load dataset
    split = "train" if train else ("val" if args.dataset == "imagenet" else "test")
    dataset = dataset(
        root=_path.join(args.dataset_root, args.dataset),
        split=split,
        transform="auto",
        target_transform="auto",
        download=True,
        **config,
    )
    return dataset


def load_model(
    args: Namespace,
    config: Config,
) -> Module:
    # Load model
    model = lib.models.load(
        name=args.model,
        num_classes=config["num_classes"],
        image_size=config["crop_size"],
        num_channels=3,
        load_from="timm",
    )
    # Replace layers
    if args.replace:
        replacements = args.replace.split(",")
        for replacement in replacements:
            _from, _to = replacement.split(":")
            model = getattr(lib.models.replace, f"{_from}_with_{_to}")(model)
            args.model += f"_{_to}"
    return model


def load_trainer(
    args: Namespace,
    config: Config,
) -> Trainer:
    trainer = lib.Trainer(
        log_root=args.log_root,
        name=f"{args.dataset}/{args.model}",
        version=args.version,
        **config,
    )
    return trainer


def main(args: Namespace):
    config = _dicts.load(args.cfg_path)
    config["devices"] = _cuda.set_visible_devices(args.device)
    train_dataset = load_dataset(args, config, train=True)
    val_dataset = load_dataset(args, config, train=False)
    model = load_model(args, config)
    trainer = load_trainer(args, config)
    trainer.run(train_dataset, val_dataset, model, args.resume)
