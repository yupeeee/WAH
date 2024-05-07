"""
e.g., python travel_cifar10.py --model resnet50 --cuda
"""
import wah

CIFAR10_ROOT = wah.path.join(".", "dataset")    # directory to download CIFAR-10 dataset
CKPT_ROOT = wah.path.join(".", "logs")          # directory where model checkpoints (i.e., weights) are saved
TRAVEL_ROOT = wah.path.join(".", "travel_res")  # directory to save travel results


if __name__ == "__main__":
    parser = wah.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tag", type=str, required=False, default="base")
    parser.add_argument("--portion", type=float, required=False, default=1.)
    parser.add_argument("--config", type=str, required=False, default="travel_cfg.yaml")
    parser.add_argument("--cuda", action="store_true", required=False, default=False)
    args = parser.parse_args()

    # load config
    config = wah.load_config(args.config)

    # load dataset/dataloader
    dataset = wah.CIFAR10(
        root=CIFAR10_ROOT,
        split="test",
        transform="tt",
        target_transform=None,
        download=True,
    )
    normalize = dataset.NORMALIZE

    if args.portion < 1:
        dataset = wah.portion_dataset(dataset, args.portion)

    dataloader = wah.load_dataloader(
        dataset=dataset,
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

    # load weights
    train_id = "cifar10"
    train_id += f"x{args.portion}" if args.portion < 1. else ""
    train_id += f"-{args.model}"

    ckpt_dir = wah.path.join(CKPT_ROOT, train_id, args.tag, "checkpoints")
    ckpt_fname = wah.path.ls(ckpt_dir, fext="ckpt")[-1]

    wah.load_state_dict(
        model=model,
        state_dict_path=wah.path.join(ckpt_dir, ckpt_fname),
        map_location=wah.device("cuda" if args.cuda else "cpu"),
    )
    model = wah.add_preprocess(model, preprocess=normalize)
    model.eval()

    # travel
    travel_id = f"{config['travel']['method']}_travel-cifar10"
    travel_id += f"x{args.portion}" if args.portion < 1. else ""
    travel_id += f"-{args.model}-{args.tag}"

    traveler = wah.Traveler(
        model,
        seed=config["seed"],
        use_cuda=args.cuda,
        **config["travel"],
    )
    travel_res = traveler.travel(dataloader, verbose=True)

    wah.dictionary.save_in_csv(
        dictionary=travel_res,
        save_dir=wah.path.join(TRAVEL_ROOT, train_id, args.tag),
        save_name=travel_id,
        index_col="gt",
    )
