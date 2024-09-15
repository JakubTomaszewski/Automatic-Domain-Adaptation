from .visa import VisaDataset, VISA_ROOT, VISA_CLS_NAMES
from .mvtec import MVTecDataset, MVTEC_ROOT, MVTEC_CLS_NAMES
from .brain_mri import BrainMRIDataset, BrainMRI_ROOT, BrainMRI_CLS_NAMES
from .headct import HEADCTDataset, HEADCT_ROOT, HEADCT_CLS_NAMES


dataset_dict = {
    "mvtec": (MVTecDataset, MVTEC_ROOT, MVTEC_CLS_NAMES),
    "visa": (VisaDataset, VISA_ROOT, VISA_CLS_NAMES),
    "brain_mri": (BrainMRIDataset, BrainMRI_ROOT, BrainMRI_CLS_NAMES),
    "headct": (HEADCTDataset, HEADCT_ROOT, HEADCT_CLS_NAMES),
}


def get_dataset(dataset_name: str, transforms: callable = None, is_test: bool = False):
    dataset, dataset_root_dir, dataset_class_names = dataset_dict[dataset_name]
    return dataset(dataset_root_dir, dataset_class_names, transforms, is_test)