import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR


"""dataset source: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"""
VISA_CLS_NAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

VISA_ROOT = os.path.join(DATA_ROOT_DIR, "VisA_20220922")


class VisaDataset(BaseDataset):
    def __init__(self,
                 root_dir: str = VISA_ROOT,
                 class_names: list[str] = VISA_CLS_NAMES,
                 meta_file: str = "meta_split.json",
                 transforms: callable = None,
                 is_test: bool = False,
                 ) -> None:
        super(VisaDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            is_test=is_test,
        )
