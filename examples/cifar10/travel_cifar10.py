"""
e.g., python travel_cifar10.py --model resnet50
"""
import argparse
import os

from torchvision import models

import wah

CIFAR10_ROOT = os.path.join(".", "dataset") # directory to download CIFAR-10 dataset
CKPT_ROOT = os.path.join(".", "weights")    # directory where model checkpoints (i.e., weights) are saved

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
    kwargs = {
        "weights": None,
        "num_classes": config["num_classes"],
    }
    if "vit" in args.model:
        kwargs["image_size"] = 32

    model = getattr(models, args.model)(**kwargs)
    wah.load_state_dict(
        model=model,
        state_dict_path=os.path.join(CKPT_ROOT, f"{args.model}.ckpt"),
        map_location=f"cuda:{','.join([str(d) for d in config['gpu']])}",
    )
    model = wah.add_preprocess(model, preprocess=normalize)
    model.eval()

    # travel
    travel_id = f"{config['travel']['method']}_travel-cifar10"
    travel_id += f"x{args.portion}" if args.portion < 1. else ""
    travel_id += f"-{args.model}"

    traveler = wah.Traveler(
        model,
        seed=config["seed"],
        use_cuda=True if "gpu" in config.keys() else False,
        **config["travel"],
    )
    travel_res = traveler.travel(dataloader, verbose=True)

    wah.save_dict_in_csv(
        dictionary=travel_res,
        save_dir=".",
        save_name=travel_id,
        index_col="gt",
    )
