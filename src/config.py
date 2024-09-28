import torch

from argparse import ArgumentParser
from utils import available_torch_device


DATA_ROOT_DIR = './data'


def parse_args() -> dict:
    parser = ArgumentParser()

    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for the random number generator (default: 42)")

    # Data
    parser.add_argument("--meta_file", type=str, default="meta_split.json",
                        help="The name of the dataset meta file containing paths to the train images (default: meta_split.json)")
    parser.add_argument("--train_dataset", type=str, default="mvtec",
                        choices=["mvtec", "visa", "brain_mri", "headct", "flowers"],
                        help="The name of the dataset to use (default: mvtec)")
    parser.add_argument("--val_dataset", type=str, default="mvtec",
                        choices=["mvtec", "visa", "brain_mri", "headct", "flowers"],
                        help="The name of the dataset to use (default: mvtec)")

    # Model
    parser.add_argument("--dinov2_backbone", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
                                 # with register tokens
                                 "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
                                 ])

    # Training
    parser.add_argument("--output_dir", type=str, default="./training_logs",
                        help="The directory to save the model and logs (default: ./output)")
    parser.add_argument('--device', type=available_torch_device, default='cuda',
                        choices=[torch.device('cpu'), torch.device('cuda'), torch.device('mps')],
                        help='Device used for computation (default: cpu)')
    parser.add_argument("--epochs", type=int, default=10,
                        help="The number of epochs to train (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The batch size for training (default: 32)")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="The batch size for evaluation (default: 32)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help="The learning rate for training (default: 1e-3)")
    parser.add_argument("--img_size", type=tuple, default=(224, 224),
                        help="The size of the images (default: (224, 224))")
    parser.add_argument("--weight_classes", action="store_true", default=False,
                        help="Whether to weight the classes in the loss function (default: False)")
    
    return parser.parse_args()
