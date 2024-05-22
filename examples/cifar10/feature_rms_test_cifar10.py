"""
e.g., python feature_rms_test_cifar10.py --model resnet50 --version base
"""

import wah

CIFAR10_ROOT = wah.path.join("F:/", "datasets", "cifar10")
CKPT_ROOT = wah.path.join("F:/", "logs")
RES_ROOT = wah.path.join("F:/", "feature_rms")


if __name__ == "__main__":
    parser = wah.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--portion", type=float, required=False, default=1.0)
    args = parser.parse_args()

    # load config
    config_path = wah.path.join(".", "cfgs", "train", f"{args.version}.yaml")
    config = wah.config.load(config_path)
    epochs = config["epochs"]

    # load dataset/dataloader
    dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="test",
        transform="auto",
        target_transform=None,
        download=True,
    )
    use_cuda = True if "gpu" in config["devices"] else False

    if args.portion < 1:
        dataset = wah.portion_dataset(dataset, args.portion)

    # compute
    test = wah.FeatureRMSTest(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_cuda=use_cuda,
    )

    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1.0 else ""
    train_id += f"/{args.model}/{args.version}"

    ckpt_dir = wah.path.join(CKPT_ROOT, train_id, "checkpoints")

    for epoch in range(epochs):
        # load model
        model = wah.load_timm_model(
            name=args.model,
            pretrained=False,
            num_classes=config["num_classes"],
            image_size=32,
            num_channels=3,
        )
        wah.load_state_dict(
            model=model,
            state_dict_path=wah.path.join(ckpt_dir, f"epoch={epoch}.ckpt"),
            map_location=wah.device("cuda" if use_cuda else "cpu"),
        )
        model.eval()

        # feature rms test
        feature_rms = test(model, dataset, verbose=True, desc=f"epoch={epoch}")

        wah.dictionary.save_in_csv(
            dictionary=feature_rms,
            save_dir=wah.path.join(RES_ROOT, train_id),
            save_name=f"epoch={epoch}",
            index_col=None,
        )
