import torch
import tqdm

from ...data.load import load_dataloader
from ...typing import (
    Dataset,
    Device,
    Dict,
    List,
    Literal,
    Module,
    Optional,
    Tensor,
)
from .utils import DirectionGenerator

__all__ = [
    "LinearityTest",
]


def move_x(
    x: Tensor,
    d: Tensor,
    eps: float,
    bound: bool = True,
) -> Tensor:
    _x = x + d * eps

    if bound:
        _x = _x.clamp(0, 1)

    return _x


def compute_cossim(
    model: Module,
    data: Tensor,
    directions: Tensor,
    eps: float,
    delta: float,
    device: Device,
    bound: bool = True,
) -> List[float]:
    batch_size = len(data)
    nzd = 1.0e-7  # for nonzero division

    model.to(device)
    data = data.to(device)
    directions = directions.to(device)

    # move data along given directions
    data_eps = move_x(data, directions, eps, bound)
    data_eps_l = move_x(data, directions, eps - delta, bound)
    data_eps_r = move_x(data, directions, eps + delta, bound)

    # compute features
    with torch.no_grad():
        # model is timm model
        if hasattr(model, "forward_features"):
            preprocess = torch.nn.Identity()
            feature_extractor = model.forward_features
        # model is timm model w/ preprocess added
        else:
            preprocess = model.preprocess
            feature_extractor = model.model.forward_features

        out_eps: Tensor = feature_extractor(preprocess(data_eps))
        out_eps_l: Tensor = feature_extractor(preprocess(data_eps_l))
        out_eps_r: Tensor = feature_extractor(preprocess(data_eps_r))

    # compute/normalize movement vectors
    vl = (out_eps_l - out_eps).reshape(batch_size, -1)
    vr = (out_eps_r - out_eps).reshape(batch_size, -1)

    vl = vl / torch.norm(vl, p=2, dim=-1, keepdim=True) + nzd
    vr = vr / torch.norm(vr, p=2, dim=-1, keepdim=True) + nzd

    # compute cosine similiarities
    cossims = torch.sum(-vl * vr, dim=-1)

    return [float(cossim) for cossim in cossims.to(torch.device("cpu"))]


class LinearityTest:
    def __init__(
        self,
        max_eps: float,
        num_steps: int,
        method: Literal["fgsm",] = "fgsm",
        batch_size: int = 1,
        num_workers: int = 0,
        delta: float = 1.0e-3,
        seed: Optional[int] = 0,
        bound: bool = True,
        use_cuda: bool = False,
    ) -> None:
        if max_eps == 0:
            self.epsilons = [
                0.0,
            ]
        else:
            self.epsilons = [
                float(eps) for eps in torch.linspace(-max_eps, max_eps, num_steps)
            ]
        self.method = method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.delta = delta
        self.seed = seed
        self.bound = bound
        self.use_cuda = use_cuda
        self.device: Device = torch.device("cuda" if use_cuda else "cpu")

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        verbose: bool = False,
    ) -> Dict[float, List[float]]:
        # load dataloader, direction_generator
        dataloader = load_dataloader(
            dataset=dataset,
            config={
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
            },
            train=False,
        )
        direction_generator = DirectionGenerator(
            model=model,
            method=self.method,
            seed=self.seed,
            use_cuda=self.use_cuda,
        )

        # init cossims
        cossims: Dict[float, List[float]] = dict()

        for eps in self.epsilons:
            cossims[eps] = []

        if verbose:
            print(
                f"Linearity test (method={self.method}) of {len(dataloader.dataset)} data"
            )

        for eps_idx, eps in enumerate(self.epsilons):
            for data, targets in tqdm.tqdm(
                dataloader,
                desc=f"[{eps_idx + 1}/{len(self.epsilons)}] eps={eps}",
                disable=not verbose,
            ):
                directions = direction_generator(data, targets)

                cossims[eps] += compute_cossim(
                    model=model,
                    data=data,
                    directions=directions,
                    eps=eps,
                    delta=self.delta,
                    device=self.device,
                    bound=self.bound,
                )

        return cossims
