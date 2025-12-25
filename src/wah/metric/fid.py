import os
from typing import Callable, Literal, Optional, Union

import torch
from cleanfid import fid as _fid

__all__ = [
    "FID",
]


class FID:
    def __init__(
        self,
        metric: Literal["fid", "kid"] = "fid",
        feature_extractor: Union[
            Literal["inception_v3", "clip_vit_b_32"], Callable
        ] = "inception_v3",
        batch_size: int = 32,
        num_workers: int = 12,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        dataset_name: str = "FFHQ",
        dataset_split: str = "train",
        dataset_res: int = 1024,
        num_gen: int = 50_000,
        z_dim: int = 512,
        use_dataparallel: bool = True,
        custom_image_tranform: Optional[Callable] = None,
        custom_fn_resize: Optional[Callable] = None,
    ) -> None:
        """
        ### Args
            - `metric` (Literal["fid", "kid"]): Metric to compute. Supports "fid" (default) and "kid".
            - `feature_extractor` (Union[Literal["inception_v3", "clip_vit_b_32"], Callable]): Model to use for FID feature extraction. Supports "inception_v3" (default), "clip_vit_b_32", and a custom callable function.
            - `batch_size` (int): Batch size for processing images. Default is 32.
            - `num_workers` (int): Number of worker threads for DataLoader. Default is 12.
            - `device` (torch.device): The device to run the computation on. Default is torch.device("cuda").
            - `verbose` (bool): If True, print progress and debug information. Default is True.
            - `dataset_name` (str): Name of the dataset for reference statistics. Default is "FFHQ".
            - `dataset_split` (str): The dataset split to use. Default is "train".
            - `dataset_res` (int): Resolution of the dataset images. Default is 1024.
            - `num_gen` (int): Number of images to generate for computing FID. Default is 50,000.
            - `z_dim` (int): Dimensionality of the latent space for generation. Default is 512.
            - `use_dataparallel` (bool): Whether to use DataParallel for feature extraction. Default is True.
            - `custom_image_tranform` (Optional[Callable]): Custom transformation function for images for FID computation. Default is None.
            - `custom_fn_resize` (Optional[Callable]): Custom function for resizing images for FID computation. Default is None.
        """
        self.metric = metric
        self.fid = _fid

        self.kwargs = {
            "mode": "clean",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device,
            "verbose": verbose,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "dataset_res": dataset_res,
            "num_gen": num_gen,
            "z_dim": z_dim,
            "use_dataparallel": use_dataparallel,
        }

        if self.metric == "fid":
            self.kwargs["model_name"] = (
                feature_extractor
                if isinstance(feature_extractor, str)
                else "inception_v3"
            )
            self.kwargs["custom_feat_extractor"] = (
                feature_extractor if not isinstance(feature_extractor, str) else None
            )
            self.kwargs["custom_image_tranform"] = custom_image_tranform
            self.kwargs["custom_fn_resize"] = custom_fn_resize

    def to(self, device: torch.device) -> "FID":
        self.kwargs["device"] = device
        return self

    def __call__(
        self,
        img_dir1: os.PathLike = None,
        img_dir2: os.PathLike = None,
        img_generator: Callable = None,
    ) -> float:
        """
        Compute the Fréchet Inception Distance (FID) or Kernel Inception Distance (KID) between two sets of images, or between generated images and a predefined dataset.

        ### Args
            - `img_dir1` (os.PathLike): Directory containing the first set of images.
            - `img_dir2` (os.PathLike): Directory containing the second set of images.
            - `img_generator` (Callable): A generator function for creating images.

        ### Returns
            - `float`: The computed FID or KID score.

        ### Raises:
            - `ValueError`: If an invalid combination of directories and models is entered.
        """
        return getattr(self.fid, f"compute_{self.metric}")(
            fdir1=img_dir1,
            fdir2=img_dir2,
            gen=img_generator,
            **self.kwargs,
        )
