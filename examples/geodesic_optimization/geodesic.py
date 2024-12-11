import wah

if __name__ == "__main__":
    parser = wah.utils.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=False, default="energy")
    parser.add_argument("--step", type=int, required=False, default=10)
    parser.add_argument("--lr", type=float, required=False, default=0.01)
    parser.add_argument("--iteration", type=int, required=False, default=100)
    parser.add_argument("--device", type=str, required=False, default="cuda")
    args = parser.parse_args()

    # load data
    dataset = wah.classification.datasets.ImageNet(
        root="./datasets/imagenet",
        split="val",
        transform="auto",
        target_transform="auto",
        return_data_only=True,
        download=True,
    )
    denormalize = wah.classification.datasets.DeNormalize(
        mean=dataset.MEAN,
        std=dataset.STD,
    )
    dataset = wah.classification.datasets.portion_dataset(
        dataset=dataset,
        portion=1 / 50,  # 1 image per class
        balanced=True,
        random_sample=False,
    )
    x1, x2 = dataset[208], dataset[281]  # x1: Labrador retriever, x2: tabby cat

    # load model
    model = wah.classification.models.load_model(
        name=args.model,
        weights="auto",
    )
    feature_extractor = wah.classification.models.FeatureExtractor(
        model=model,
        penultimate_only=True,
    )

    # optimize geodesic path
    geodesic, losses = wah.riemann.optimize_geodesic(
        model=feature_extractor,
        x1=x1,
        x2=x2,
        strategy=args.strategy,
        steps=args.step,
        lr=args.lr,
        iterations=args.iteration,
        device=args.device,
        verbose=True,
    )
    geodesic = denormalize(geodesic).clip(0, 1)

    if len(losses.shape) == 1:
        losses = losses.unsqueeze(dim=-1)
    losses_dict = {i + 1: losses[:, i] for i in range(losses.size(-1))}

    t_values = wah.torch.linspace(0, 1, 10)
    linear_interpolation = wah.torch.stack(
        [x1 * (1 - t) + x2 * t for t in t_values], dim=0
    )
    linear_interpolation = denormalize(linear_interpolation).clip(0, 1)

    paths = wah.torch.cat([linear_interpolation, geodesic], dim=0)

    # save results
    save_dir = wah.path.join(
        "./geodesic/imagenet",
        args.model,
    )
    data_id = f"{args.strategy}_{args.model}_step_{args.step}_lr_{args.lr}_iteration_{args.iteration}"
    wah.path.mkdir(wah.path.join(save_dir, "data"))

    geodesic_plot = wah._plot.ImShow(
        height=2,
        width=args.step,
        scale=1,
        no_axis=True,
    )
    geodesic_plot.plot(paths)
    geodesic_plot.save(
        wah.path.join(
            save_dir,
            f"geodesic_{data_id}.png",
        )
    )
    wah.torch.save(
        geodesic,
        wah.path.join(
            save_dir,
            "data",
            f"geodesic_{data_id}.pt",
        ),
    )

    losses_plot = wah._plot.DistPlot2D(
        figsize=(3, 3),
        fontsize=15,
        xlabel="step",
        xlim=(0, args.step),
        xticks=wah.np.arange(args.step + 1),
        ylabel="∇E" if args.strategy == "energy" else "Path length",
        grid_alpha=0.25,
    )
    losses_plot.plot(losses_dict)
    losses_plot.save(
        wah.path.join(
            save_dir,
            f"loss_{data_id}.png",
        )
    )
    wah.torch.save(
        geodesic,
        wah.path.join(
            save_dir,
            "data",
            f"loss_{data_id}.pt",
        ),
    )
