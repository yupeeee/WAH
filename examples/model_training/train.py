import wah

wah.utils.disable_ssl_verification()

if __name__ == "__main__":
    parser = wah.utils.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--portion", type=float, required=False, default=1.0)
    parser.add_argument("--resume", type=str, required=False, default=None)
    args = parser.parse_args()

    # load config
    config = wah.utils.load_yaml_to_dict("config.yaml")

    # load dataset/dataloader
    train_dataset = wah.classification.datasets.CIFAR10(
        root="./datasets/cifar10",
        split="train",
        transform="auto",
        target_transform="auto",
        download=True,
    )
    val_dataset = wah.classification.datasets.CIFAR10(
        root="./datasets/cifar10",
        split="test",
        transform="auto",
        target_transform="auto",
        download=True,
    )

    if args.portion < 1:
        train_dataset = wah.classification.datasets.portion_dataset(
            train_dataset, args.portion
        )
        val_dataset = wah.classification.datasets.portion_dataset(
            val_dataset, args.portion
        )

    train_dataloader = wah.classification.datasets.to_dataloader(
        dataset=train_dataset,
        train=True,
        **config,
    )
    val_dataloader = wah.classification.datasets.to_dataloader(
        dataset=val_dataset,
        train=False,
        **config,
    )

    # load model
    model = wah.classification.models.load_model(
        name=args.model,
        weights=None,
        num_classes=config["num_classes"],
        image_size=32,
        num_channels=3,
    )
    model = wah.classification.train.Wrapper(model, config)

    # train
    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1.0 else ""
    train_id += f"/{args.model}"

    trainer = wah.classification.train.load_trainer(
        config=config,
        save_dir="./logs",
        name=train_id,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=(
            wah.path.join(
                "./logs",
                train_id,
                args.resume,
                "checkpoints",
                "last.ckpt",
            )
            if args.resume is not None
            else None
        ),
    )
