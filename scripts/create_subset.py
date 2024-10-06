"""
Generates a meta json file for a subset of the dataset.
The number of training samples is specified by the user.
The test set remains the same as the original dataset.
"""

import os
import json
import numpy as np

from argparse import ArgumentParser


if __name__ == '__main__':
    # load the meta_split.json file
    # select N samples for each class
    # copy the test samples to the new meta_subset.json file
    # save the new meta_subset.json file

    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--samples_per_class", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    subset_meta = {"train": {}, "test": {}}

    with open(os.path.join(args.dataset_dir, "meta_split.json"), "r") as f:
        meta = json.load(f)
    
    for class_name, class_data in meta["train"].items():
        if len(class_data) < args.samples_per_class:
            raise ValueError(f"Class {class_name} has less than {args.samples_per_class} samples. Specify a lower number.")
        subset_meta["train"][class_name] = np.array(class_data)[np.random.choice(len(class_data), args.samples_per_class, replace=False)].tolist()
    
    subset_meta["test"] = meta["test"]
    
    with open(os.path.join(args.dataset_dir, f"meta_subset_{args.samples_per_class}.json"), "w") as f:
        json.dump(subset_meta, f)

    print(f"New meta_subset.json file saved to {os.path.join(args.dataset_dir, f'meta_subset_{args.samples_per_class}.json')}")
