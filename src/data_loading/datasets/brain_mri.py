import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR


"""dataset source: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection"""
BRAINMRI_CLS_NAMES = [
    "brain_mri",
]

BRAINMRI_ROOT = os.path.join(DATA_ROOT_DIR, "BrainMRI")


class BrainMRIDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str = BRAINMRI_ROOT,
        class_names: list[str] = BRAINMRI_CLS_NAMES,
        meta_file: str = "meta_split.json",
        transforms: callable = None,
        is_test: bool = False,
    ) -> None:
        super(BrainMRIDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            meta_file=meta_file,
            transforms=transforms,
            is_test=is_test,
        )
