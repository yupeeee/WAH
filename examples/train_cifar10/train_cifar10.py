"""
TRAIN:              python train_cifar10.py --model MODEL_TO_TRAIN
CHECK TRAIN LOGS:   tensorboard --logdir logs
"""
import argparse
import os

from torchvision.datasets import CIFAR10
from torchvision import models
import torchvision.transforms as tf

import wah

CIFAR10_ROOT = os.path.join(".", "dataset")  # directory to download CIFAR-10 dataset
TRAIN_LOG_ROOT = os.path.join(".", "logs")  # directory to save train logs
EVERY_N_EPOCHS = 10  # checkpoints will be saved every 10 epochs


def load_train_dataset(root):
    train_transform = tf.Compose([
        tf.RandomHorizontalFlip(),
        tf.RandomCrop(32, 4),
        tf.ToTensor(),
        tf.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    return CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        target_transform=None,
        download=True,
    )


def load_val_dataset(root):
    val_transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    return CIFAR10(
        root=root,
        train=False,
        transform=val_transform,
        target_transform=None,
        download=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    if args.model not in models.list_models():
        raise AttributeError(
            f"PyTorch does not support {args.model}. "
            f"Check torchvision.models.list_models() for supported models."
        )

    # load config
    config = wah.load_config("config.yaml")

    # load dataset/dataloader
    train_dataset = load_train_dataset(CIFAR10_ROOT)
    val_dataset = load_val_dataset(CIFAR10_ROOT)

    train_dataloader = wah.load_dataloader(train_dataset, config, shuffle=True)
    val_dataloader = wah.load_dataloader(val_dataset, config, shuffle=False)

    # load model
    model = getattr(models, args.model)(weights=None, num_classes=10)
    model = wah.Wrapper(model, config)

    # train!
    trainer = wah.load_trainer(
        config=config,
        save_dir=TRAIN_LOG_ROOT,
        name=f"cifar10-{args.model}",
        every_n_epochs=EVERY_N_EPOCHS,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
