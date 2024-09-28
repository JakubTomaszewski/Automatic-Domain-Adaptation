import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR


"""dataset source: https://paperswithcode.com/dataset/mvtecad"""
MVTEC_CLS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

MVTEC_ROOT = os.path.join(DATA_ROOT_DIR, "mvtec_anomaly_detection")


class MVTecDataset(BaseDataset):
    def __init__(self,
                 root_dir: str = MVTEC_ROOT,
                 class_names: list[str] = MVTEC_CLS_NAMES,
                 meta_file: str = "meta_split.json",
                 transforms: callable = None,
                 is_test: bool = False,
                 ) -> None:
        super(MVTecDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            is_test=is_test,
        )