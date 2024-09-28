import json

from sklearn.model_selection import train_test_split
from argparse import ArgumentParser



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--metadata_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.metadata_file_path, "r") as f:
        metadata = json.load(f)["test"]

    metadata_split = {
        "train": {},
        "test": {}
    }
    for class_id, data in metadata.items():
        labels = [data_point["label"] for data_point in data]
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
        
        print(f"Class: {class_id}, num positive labels train: {sum(y_train)}, num negative labels train: {len(y_train) - sum(y_train)}")
        print(f"Class: {class_id}, num positive labels test: {sum(y_test)}, num negative labels test: {len(y_test) - sum(y_test)}")
        
        metadata_split["train"][class_id] = X_train
        metadata_split["test"][class_id] = X_test
        
    with open(args.output_file_path, "w") as f:
        json.dump(metadata_split, f)
