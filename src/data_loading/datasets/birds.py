import os
import json

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR

BIRDS_ROOT = os.path.join(DATA_ROOT_DIR, "birds")

BIRDS_CLS_NAMES = json.load(open(os.path.join(BIRDS_ROOT, "classes.json")))


class BirdsDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str = BIRDS_ROOT,
        class_names: list[str] = BIRDS_CLS_NAMES,
        meta_file: str = "meta_split.json",
        transforms: callable = None,
        augmentations: callable = None,
        is_test: bool = False,
    ) -> None:
        super(BirdsDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            augmentations=augmentations,
            is_test=is_test,
        )
