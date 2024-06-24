import wah


if __name__ == "__main__":
    parser = wah.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    parser.add_argument("--cuda", action="store_true", required=False, default=False)
    args = parser.parse_args()

    dataset = wah.datasets.ImageNetVal(
        root="F:/datasets/imagenet",
        transform="auto",
        target_transform="auto",
        download=True,
    )

    model = wah.models.load_model(
        name=args.model,
        weights="IMAGENET1K_V1",
        load_from="torchvision",
    ).eval()

    test = wah.models.AccuracyTest(
        top_k=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cuda=args.cuda,
        devices="0",
    )
    acc1 = test(
        model=model,
        dataset=dataset,
        verbose=True,
        desc=f"Acc@1 of {args.model} on ImageNet",
    )
    print(f"Acc@1 of {args.model} on ImageNet: {acc1 * 100:.2f}%")
