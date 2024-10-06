import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR

CIFAR10_CLS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

CIFAR10_ROOT = os.path.join(DATA_ROOT_DIR, "cifar10")


class Cifar10Dataset(BaseDataset):
    def __init__(self,
                 root_dir: str = CIFAR10_ROOT,
                 class_names: list[str] = CIFAR10_CLS_NAMES,
                 meta_file: str = "meta_split.json",
                 transforms: callable = None,
                 augmentations: callable = None,
                 is_test: bool = False,
                 ) -> None:
        super(Cifar10Dataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            augmentations=augmentations,
            is_test=is_test,
        )
