import torch
import tqdm

from ...data.load import load_dataloader
from ...typing import (
    Dataset,
    Device,
    Dict,
    List,
    Module,
    Optional,
)
from ..train import track

__all__ = [
    "FeatureRMSTest",
]


class FeatureRMSTest:
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cuda = use_cuda
        self.device: Device = torch.device("cuda" if use_cuda else "cpu")

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        dataloader = load_dataloader(
            dataset=dataset,
            config={
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
            },
            train=False,
        )

        model.to(self.device)
        feature_extractor, _, feature_rms = track.feature_rms.init(model)

        for data, _ in tqdm.tqdm(
            dataloader,
            desc=(
                f"Feature RMS test of {len(dataloader.dataset)} data"
                if desc is None
                else desc
            ),
            disable=not verbose,
        ):
            data = data.to(self.device)

            if feature_extractor.checked_layers is False:
                with torch.no_grad():
                    _ = feature_extractor(data)

                feature_rms = dict(
                    (i_layer, [])
                    for i_layer in feature_extractor.feature_layers.values()
                )

            track.feature_rms.compute(
                data=data,
                feature_extractor=feature_extractor,
                feature_rms_dict=feature_rms,
            )

        for k, v in feature_rms.items():
            feature_rms[k] = [
                float(rms) for rms in torch.cat(v).to(torch.device("cpu"))
            ]

        return feature_rms
