from torchmetrics.classification import MulticlassAccuracy

__all__ = [
    "Acc1",
    "Acc5",
]


class Acc1(MulticlassAccuracy):
    label = "acc@1"

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=1,
        )


class Acc5(MulticlassAccuracy):
    label = "acc@5"

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=5,
        )
