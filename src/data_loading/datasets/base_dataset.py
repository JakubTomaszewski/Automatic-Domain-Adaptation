import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

from typing import Union


class DataSolver:
    def __init__(self, root_dir, class_names, meta_file):
        self.root_dir = root_dir
        self.class_names = class_names
        self.path = os.path.join(root_dir, meta_file)

    def run(self):
        with open(self.path, "r") as f:
            info = json.load(f)

        info_required = dict(train={}, test={})
        for cls in self.class_names:
            for k in info.keys():
                info_required[k][cls] = info[k][cls]

        return info_required


class BaseDataset(Dataset):
    def __init__(self, 
                 root_dir: Union[str, os.PathLike], 
                 class_names: list[str], 
                 meta_file: str = "meta_split.json",
                 transforms: callable = None,
                 is_test: bool  = False
                 ) -> None:
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_all = []
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.is_test = is_test

        solver = DataSolver(root_dir, class_names, meta_file)
        meta_info = solver.run()

        self.meta_info = meta_info["train"] if not self.is_test else meta_info["test"]
        for cls_name in self.class_names:
            self.data_all.extend(self.meta_info[cls_name])

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        cls_name = data["cls_name"]
        label = data["label"]

        img_path = os.path.join(self.root_dir, data["img_path"])
        img = self.load_image(img_path)

        if self.transforms is not None:
            img = self.transforms(img)
        return {
            "x": img,
            "label": label,
            "cls_name": cls_name,
            "img_path": img_path,
        }

    def get_class_names(self):
        return self.class_names

    def load_image(self, img_path):
        """Loads images using torch read_image function"""
        img = Image.open(img_path).convert("RGB")
        return img

    def get_class_weights(self):
        class_weights = []
        labels = [data_point["label"] for data_point in self.data_all]
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        return torch.Tensor(class_weights)
