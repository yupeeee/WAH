"""
e.g., python eval_cifar10.py --model resnet50
"""

import wah

CIFAR10_ROOT = wah.path.join("F:/", "dataset", "cifar10")
CKPT_ROOT = wah.path.join("F:/", "logs")


if __name__ == "__main__":
    parser = wah.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tag", type=str, required=False, default="base")
    parser.add_argument("--portion", type=float, required=False, default=1.0)
    parser.add_argument("--config", type=str, required=False, default="config.yaml")
    args = parser.parse_args()

    # set start method for ddp
    wah.dist.set_start_method()

    # load config
    config = wah.config.load(args.config)
    use_cuda = True if "gpu" in config.keys() else False

    # load dataset/dataloader
    train_dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="train",
        transform="test",
        target_transform="auto",
        download=True,
    )
    test_dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="test",
        transform="test",
        target_transform="auto",
        download=True,
    )

    if args.portion < 1:
        train_dataset = wah.portion_dataset(train_dataset, args.portion)
        test_dataset = wah.portion_dataset(test_dataset, args.portion)

    # load model
    model = wah.load_timm_model(
        name=args.model,
        pretrained=False,
        num_classes=10,
        image_size=32,
        num_channels=3,
    )

    # load weights
    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1.0 else ""
    train_id += f"/{args.model}"

    ckpt_dir = wah.path.join(CKPT_ROOT, train_id, args.tag, "checkpoints")
    ckpt_fname = wah.path.ls(ckpt_dir, fext="ckpt")[-1]

    wah.load_state_dict(
        model=model,
        state_dict_path=wah.path.join(ckpt_dir, ckpt_fname),
        map_location=wah.device("cuda" if use_cuda else "cpu"),
    )
    model.eval()

    # evaluation: acc@1 test
    test = wah.AccuracyTest(
        top_k=1,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_cuda=use_cuda,
        devices=config["gpu"] if "gpu" in config.keys() else "auto",
    )

    train_acc1 = test(
        model=model,
        dataset=train_dataset,
        verbose=True,
        desc="Computing acc@1 on cifar10 train dataset...",
    )
    test_acc1 = test(
        model=model,
        dataset=test_dataset,
        verbose=True,
        desc="Computing acc@1 on cifar10 test dataset...",
    )

    dataset_id = train_id.split("-")[0]

    print(f"acc@1 of {args.model} on {dataset_id} (train): {train_acc1 * 100:.2f}%")
    print(f"acc@1 of {args.model} on {dataset_id} (test): {test_acc1 * 100:.2f}%")
