import wah

if __name__ == "__main__":
    parser = wah.utils.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    args = parser.parse_args()

    dataset = wah.classification.datasets.ImageNet(
        root="./datasets/imagenet",
        split="val",
        transform="auto",
        target_transform="auto",
        download=True,
    )

    model = wah.classification.models.load_model(
        name=args.model,
        weights="IMAGENET1K_V1",
        load_from="torchvision",
    )

    test = wah.classification.test.AccuracyTest(
        top_k=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        devices="auto",
    )
    acc1 = test(
        model=model,
        dataset=dataset,
    )
    print(f"Acc@1 of {args.model} on ImageNet: {acc1 * 100:.2f}%")
