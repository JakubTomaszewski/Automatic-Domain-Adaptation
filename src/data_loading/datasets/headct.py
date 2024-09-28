import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR


"""dataset source: https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage"""
HEADCT_CLS_NAMES = [
    "headct",
]

HEADCT_ROOT = os.path.join(DATA_ROOT_DIR, "HeadCT_anomaly_detection")


class HEADCTDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str = HEADCT_ROOT,
        class_names: list[str] = HEADCT_CLS_NAMES,
        meta_file: str = "meta_split.json",
        transforms: callable = None,
        is_test: bool = False,
    ) -> None:
        super(HEADCTDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            is_test=is_test,
        )
