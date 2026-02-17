import os
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset

from ....misc import dicts as _dicts
from ....misc import path as _path
from ...utils import _download_from_url, _md5_check

__all__ = [
    "_WebsterArXiv2023",
]


class _WebsterArXiv2023(Dataset):
    def __init__(
        self,
        root: os.PathLike,
        url: str,
        checksum: str,
    ) -> None:
        root = _path.clean(root)
        filename = os.path.basename(url)

        # Download dataset
        if not os.path.exists(os.path.join(root, filename)):
            _download_from_url(url, root)

        # Check checksum
        while not _md5_check(
            path=os.path.join(root, filename),
            checksum=checksum,
        ):
            print("Dataset corrupted. Redownloading dataset.")
            os.remove(os.path.join(root, filename))
            _download_from_url(url, root)

        # Load dataset
        data = _dicts.load(
            path=os.path.join(root, filename),
        )
        self.captions = data["caption"]
        self.indices = data["index"]
        # self.scores = data["scores"] <- unable to use since gen_seeds is unknown
        self.urls = data["url"]
        # self.num_duplicates = data["numdups"] <- missing in pipe = "deep_if"
        # self.edge_scores = data["edge_scores"] <- missing in pipe = "deep_if"
        # self.mse_real_gens = data["mse_real_gen"] <- unable to use since gen_seeds is unknown
        self.overfit_types = data["overfit_type"]
        # self.gen_seeds = data["gen_seeds"] <- empty values
        # self.retrieved_urls = data["retrieved_urls"] <- empty values

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "caption": self.captions[idx],
            "index": self.indices[idx],
            # "scores": self.scores[idx],
            "url": self.urls[idx],
            # "num_duplicates": self.num_duplicates[idx],
            # "edge_scores": self.edge_scores[idx],
            # "mse_real_gen": self.mse_real_gens[idx],
            "overfit_type": self.overfit_types[idx],
            # "gen_seeds": self.gen_seeds[idx],
            # "retrieved_urls": self.retrieved_urls[idx],
        }

    def get_index(self, idx: int) -> int:
        return self.indices[idx]

    def get_caption(self, index: int) -> str:
        return self.captions[self.indices.index(index)]

    # def get_score(self, index: int) -> float:
    #     return self.scores[self.indices.index(index)]

    def get_url(self, index: int) -> str:
        return self.urls[self.indices.index(index)]

    # def get_num_duplicates(self, index: int) -> int:
    #     return self.num_duplicates[self.indices.index(index)]

    # def get_edge_score(self, index: int) -> float:
    #     return self.edge_scores[self.indices.index(index)]

    # def get_mse_real_gen(self, index: int) -> float:
    #     return self.mse_real_gens[self.indices.index(index)]

    def get_overfit_type(self, index: int) -> str:
        return self.overfit_types[self.indices.index(index)]

    # def get_gen_seed(self, index: int) -> int:
    #     return self.gen_seeds[self.indices.index(index)]

    # def get_retrieved_url(self, index: int) -> str:
    #     return self.retrieved_urls[self.indices.index(index)]

    def split_indices(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        overfit_types = [self.get_overfit_type(index) for index in self.indices]

        # MATCHING VERBATIM
        MV_indices = [
            self.indices[i]
            for i in range(len(self.indices))
            if overfit_types[i] == "MV"
        ]
        # TEMPLATE DUPLICATE
        TV_indices = [
            self.indices[i]
            for i in range(len(self.indices))
            if overfit_types[i] == "TV"
        ]
        # RETRIEVED VERBATIM
        RV_indices = [
            self.indices[i]
            for i in range(len(self.indices))
            if overfit_types[i] == "RV"
        ]
        # NONE
        N_indices = [
            self.indices[i] for i in range(len(self.indices)) if overfit_types[i] == "N"
        ]

        return (
            MV_indices,
            TV_indices,
            RV_indices,
            N_indices,
        )
