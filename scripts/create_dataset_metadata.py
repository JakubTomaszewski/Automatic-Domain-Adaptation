"""
Generates a meta json with a particular formal based on the directory structure of the dataset.

Output json format:
{
  "train": {
    "class_name_1": [
        {
          "img_path": "path/to/image1",
          "cls_name": "class_name_1",
          "label": 0
        },
        {
            "img_path": "path/to/image2",
            "cls_name": "class_name_1",
            "label": 1
        }
    ],
    "class_name_2": [
        {
            "img_path": "path/to/image3",
            "cls_name": "class_name_2",
            "label": 0
        }
    ]
    },
    "test": {
        "class_name_1": [
            {
                "img_path": "path/to/image4",
                "cls_name": "class_name_1",
                "label": 0
            }
        ],
        "class_name_2": [
            {
                "img_path": "path/to/image5",
                "cls_name": "class_name_2",
                "label": 1
            }
        ]
    }
}

"""

import os
import json

from sklearn.model_selection import train_test_split
from collections import defaultdict
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--data_folder", type=str, default="train")
    parser.add_argument("--train_test_split", action="store_true")
    
    args = parser.parse_args()
    
    metadata = {
        "train": {},
        "test": {}
    }
    
    data_path = os.path.join(args.dataset_dir, args.data_folder)
    class_names = os.listdir(data_path)
    
    for class_id, class_name in enumerate(class_names):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Skipping {class_dir} since it is not a directory")
            continue
        
        class_dict = defaultdict(list)
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(args.data_folder, class_name, img_name)
            img_info = {
                "img_path": img_path,
                "cls_name": class_name,
                "label": class_id
            }
            class_dict[class_name].append(img_info)
        metadata["train"][class_name] = class_dict[class_name]

    with open(os.path.join(args.dataset_dir, "meta.json"), "w") as f:
        json.dump(metadata, f)

    print(f"meta.json file saved to {os.path.join(args.dataset_dir, 'meta.json')}")

    if args.train_test_split:
        print("Splitting the data into train and test")
        # Split the data into train and test
        metadata_split = {
            "train": {},
            "test": {}
        }
        for class_id, data in metadata["train"].items():
            labels = [data_point["label"] for data_point in data]
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
                    
            metadata_split["train"][class_id] = X_train
            metadata_split["test"][class_id] = X_test
            
        with open(os.path.join(args.dataset_dir, "meta_split.json"), "w") as f:
            json.dump(metadata_split, f)
        
        print(f"meta_split.json file saved to {os.path.join(args.dataset_dir, 'meta_split.json')}")
