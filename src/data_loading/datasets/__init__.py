from .visa import VisaDataset, VISA_ROOT, VISA_CLS_NAMES
from .mvtec import MVTecDataset, MVTEC_ROOT, MVTEC_CLS_NAMES
from .brain_mri import BrainMRIDataset, BRAINMRI_ROOT, BRAINMRI_CLS_NAMES
from .headct import HEADCTDataset, HEADCT_ROOT, HEADCT_CLS_NAMES
from .flowers import FlowersDataset, FLOWERS_ROOT, FLOWERS_CLS_NAMES
from .birds import BirdsDataset, BIRDS_ROOT, BIRDS_CLS_NAMES
from .cifar10 import Cifar10Dataset, CIFAR10_ROOT, CIFAR10_CLS_NAMES


dataset_dict = {
    "birds": (BirdsDataset, BIRDS_ROOT, BIRDS_CLS_NAMES),
    "mvtec": (MVTecDataset, MVTEC_ROOT, MVTEC_CLS_NAMES),
    "visa": (VisaDataset, VISA_ROOT, VISA_CLS_NAMES),
    "brain_mri": (BrainMRIDataset, BRAINMRI_ROOT, BRAINMRI_CLS_NAMES),
    "headct": (HEADCTDataset, HEADCT_ROOT, HEADCT_CLS_NAMES),
    "flowers": (FlowersDataset, FLOWERS_ROOT, FLOWERS_CLS_NAMES),
    "cifar10": (Cifar10Dataset, CIFAR10_ROOT, CIFAR10_CLS_NAMES),
}


def get_dataset(dataset_name: str, meta_file: str, transforms: callable = None, augmentations: callable = None, is_test: bool = False):
    dataset, dataset_root_dir, dataset_class_names = dataset_dict[dataset_name]
    return dataset(dataset_root_dir, dataset_class_names, meta_file, transforms, augmentations, is_test)