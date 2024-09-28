import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR

FLOWERS_CLS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

FLOWERS_ROOT = os.path.join(DATA_ROOT_DIR, "flowers")


class FlowersDataset(BaseDataset):
    def __init__(self,
                 root_dir: str = FLOWERS_ROOT,
                 class_names: list[str] = FLOWERS_CLS_NAMES,
                 meta_file: str = "meta_split.json",
                 transforms: callable = None,
                 is_test: bool = False,
                 ) -> None:
        super(FlowersDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            is_test=is_test,
        )
