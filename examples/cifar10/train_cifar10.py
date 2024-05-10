"""
TRAIN:              python train_cifar10.py --model MODEL_TO_TRAIN
CHECK TRAIN LOGS:   tensorboard --logdir logs
"""

from src import wah

CIFAR10_ROOT = wah.path.join("F:/", "datasets", "cifar10")
TRAIN_LOG_ROOT = wah.path.join("F:/", "logs")


if __name__ == "__main__":
    parser = wah.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--portion", type=float, required=False, default=1.0)
    parser.add_argument("--config", type=str, required=False, default="config.yaml")
    args = parser.parse_args()

    # load config
    config = wah.config.load(args.config)

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
        train=True,
    )
    val_dataloader = wah.load_dataloader(
        dataset=val_dataset,
        config=config,
        train=False,
    )

    # load model
    model = wah.load_timm_model(
        name=args.model,
        pretrained=False,
        num_classes=10,
        image_size=32,
        num_channels=3,
    )
    model = wah.Wrapper(model, config)

    # train
    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1.0 else ""
    train_id += f"/{args.model}"

    trainer = wah.load_trainer(
        config=config,
        save_dir=TRAIN_LOG_ROOT,
        name=train_id,
        every_n_epochs=1,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
