import os

from .base_dataset import BaseDataset
from config import DATA_ROOT_DIR


'''dataset source: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection'''
BrainMRI_CLS_NAMES = [
    'brain_mri',
]

BrainMRI_ROOT = os.path.join(DATA_ROOT_DIR, 'BrainMRI')


class BrainMRIDataset(BaseDataset):
    def __init__(self,
                 root_dir: str = BrainMRI_ROOT,
                 class_names: list[str] = BrainMRI_CLS_NAMES,
                 transforms: callable = None,
                 is_test: bool = False,
                 ) -> None:
        super(BrainMRIDataset, self).__init__(
            root_dir=root_dir,
            class_names=class_names,
            transforms=transforms,
            is_test=is_test,
        )
