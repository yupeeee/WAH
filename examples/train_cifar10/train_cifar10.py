"""
TRAIN:              python train_cifar10.py --model MODEL_TO_TRAIN
CHECK TRAIN LOGS:   tensorboard --logdir logs
"""
import argparse
import os

from torchvision import models

import wah

CIFAR10_ROOT = os.path.join(".", "dataset")  # directory to download CIFAR-10 dataset
TRAIN_LOG_ROOT = os.path.join(".", "logs")  # directory to save train logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--portion", type=float, required=False, default=1.)
    parser.add_argument("--config", type=str, required=False, default="config.yaml")
    args = parser.parse_args()

    if args.model not in models.list_models():
        raise AttributeError(
            f"PyTorch does not support {args.model}. "
            f"Check torchvision.models.list_models() for supported models."
        )

    # load config
    config = wah.load_config("config.yaml")

    # load dataset/dataloader
    train_dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="train",
        transform="auto",
        target_transform="auto",
        download=True,
    )
    val_dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="test",
        transform="auto",
        target_transform="auto",
        download=True,
    )

    if args.portion < 1:
        train_dataset = wah.portion_dataset(train_dataset, args.portion)
        val_dataset = wah.portion_dataset(val_dataset, args.portion)

    train_dataloader = wah.load_dataloader(
        dataset=train_dataset,
        config=config,
        shuffle=True,
    )
    val_dataloader = wah.load_dataloader(
        dataset=val_dataset,
        config=config,
        shuffle=False,
    )

    # load model
    kwargs = {
        "weights": None,
        "num_classes": config["num_classes"],
    }
    if "vit" in args.model:
        kwargs["image_size"] = 32

    model = getattr(models, args.model)(**kwargs)
    model = wah.Wrapper(model, config)

    # train
    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1. else ""
    train_id += f"-{args.model}"

    trainer = wah.load_trainer(
        config=config,
        save_dir=TRAIN_LOG_ROOT,
        name=train_id,
        every_n_epochs=config["epochs"],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
