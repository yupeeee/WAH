"""
e.g., python linearity_test_cifar10.py --model resnet50 --version base --method fgsm
"""

import wah

CIFAR10_ROOT = wah.path.join("F:/", "datasets", "cifar10")
CKPT_ROOT = wah.path.join("F:/", "logs")
RES_ROOT = wah.path.join("F:/", "linearity")


if __name__ == "__main__":
    parser = wah.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--portion", type=float, required=False, default=1.0)
    parser.add_argument("--method", type=str, required=False, default="fgsm")
    args = parser.parse_args()

    # load config
    config_path = wah.path.join(".", "cfgs", "linearity", f"{args.method}.yaml")
    config = wah.config.load(config_path)

    # load dataset/dataloader
    dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="test",
        transform="tt",
        target_transform=None,
        download=True,
    )
    normalize = dataset.NORMALIZE
    use_cuda = True if "gpu" in config["devices"] else False

    if args.portion < 1:
        dataset = wah.portion_dataset(dataset, args.portion)

    # load model
    model = wah.load_timm_model(
        name=args.model,
        pretrained=False,
        num_classes=config["num_classes"],
        image_size=32,
        num_channels=3,
    )

    # load weights
    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1.0 else ""
    train_id += f"/{args.model}"

    ckpt_dir = wah.path.join(CKPT_ROOT, train_id, args.version, "checkpoints")
    ckpt_fname = wah.path.ls(ckpt_dir, fext=f"ckpt")[-1]
    # ckpt_fname = [f for f in wah.path.ls(ckpt_dir, fext="ckpt") if "epoch=" in f][-1]

    wah.load_state_dict(
        model=model,
        state_dict_path=wah.path.join(ckpt_dir, ckpt_fname),
        map_location=wah.device("cuda" if use_cuda else "cpu"),
    )
    model = wah.add_preprocess(model, preprocess=normalize)
    model.eval()

    # linearity test
    travel_id = f"{args.method}_travel-cifar10"
    travel_id += f"x{args.portion}" if args.portion < 1.0 else ""
    travel_id += f"-{args.model}-{args.version}"

    test = wah.LinearityTest(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        use_cuda=use_cuda,
        **config["test"],
    )
    cossims = test(model, dataset, verbose=True)

    wah.dictionary.save_in_csv(
        dictionary=cossims,
        save_dir=wah.path.join(RES_ROOT, train_id),
        save_name=travel_id,
        index_col=None,
    )
