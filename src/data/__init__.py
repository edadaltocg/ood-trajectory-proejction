"""
Datasets module.
"""
import logging
from enum import Enum
from functools import partial
from typing import Callable, List, Optional, Type

from torch.utils.data import Dataset
from torchvision.datasets import STL10, SVHN, ImageNet, OxfordIIITPet, StanfordCars

from ..config import DATA_DIR, IMAGENET_ROOT
from .cifar_wrapper import CIFAR10Wrapped, CIFAR100Wrapped
from .constants import *
from .english_chars import EnglishChars
from .isun import iSUN
from .lsun_r_c import LSUNCroped, LSUNResized
from .mnist_wrapped import FashionMNISTWrapped, MNISTWrapped
from .mos import MOSSUN, MOSiNaturalist, MOSPlaces365
from .noise import Blobs, Gaussian, Rademacher, Uniform
from .places365 import Places365
from .textures import Textures
from .tiny_imagenet import TinyImageNet
from .tiny_imagenet_r_c import TinyImageNetCroped, TinyImageNetResized

_logger = logging.getLogger(__name__)
datasets_registry = {
    "cifar10": CIFAR10Wrapped,
    "cifar100": CIFAR100Wrapped,
    "stl10": STL10,
    "svhn": SVHN,
    "mnist": MNISTWrapped,
    "fashion_mnist": FashionMNISTWrapped,
    "english_chars": EnglishChars,
    "isun": iSUN,
    "lsun_c": LSUNCroped,
    "lsun_r": LSUNResized,
    "tiny_imagenet_c": TinyImageNetCroped,
    "tiny_imagenet_r": TinyImageNetResized,
    "tiny_imagenet": TinyImageNet,
    "textures": Textures,
    "gaussian": Gaussian,
    "uniform": Uniform,
    "blobs": Blobs,
    "rademacher": Rademacher,
    "places365": Places365,
    "mos_inaturalist": MOSiNaturalist,
    "mos_places365": MOSPlaces365,
    "mos_sun": MOSSUN,
    "imagenet": ImageNet,
    "imagenet1k": ImageNet,
    "ilsvrc2012": ImageNet,
}


def register_dataset(dataset_name: str):
    """Register a dataset on the `datasets_registry`.

    Args:
        dataset_name (str): Name of the dataset.

    Example::

        @register_dataset("my_dataset")
        class MyDataset(Dataset):
            ...

        dataset = create_dataset("my_dataset")
    """

    def register_model_cls(cls):
        if dataset_name in datasets_registry:
            raise ValueError(f"Cannot register duplicate dataset ({dataset_name})")
        datasets_registry[dataset_name] = cls
        return cls

    return register_model_cls


def create_dataset(
    dataset_name: str,
    root: str = DATA_DIR,
    split: Optional[str] = "train",
    transform: Optional[Callable] = None,
    download: Optional[bool] = True,
    **kwargs,
):
    """Create dataset factory.
    """
    try:
        if dataset_name in ["imagenet", "imagenet1k", "ilsvrc2012"]:
            return datasets_registry[dataset_name](root=IMAGENET_ROOT, split=split, transform=transform, **kwargs)
        return datasets_registry[dataset_name](root=root, split=split, transform=transform, download=download, **kwargs)
    except KeyError as e:
        _logger.error(e)
        raise ValueError("Dataset name is not specified")


def get_dataset_cls(dataset_name: str) -> Type[Dataset]:
    """Return dataset class by name.

    Args:
        dataset_name (string): Name of the dataset.

    Raises:
        ValueError: If dataset name is not available in `datasets_registry`.

    Returns:
        Dataset: Dataset class.
    """
    return datasets_registry[dataset_name]


def list_datasets() -> List[str]:
    """List of available dataset names, sorted alphabetically.

    Returns:
        list: List of available dataset names.
    """
    return sorted(list(k for k in datasets_registry.keys() if datasets_registry[k] is not None))


DatasetsRegistry = Enum("DatasetsRegistry", dict(zip(list_datasets(), list_datasets())))
