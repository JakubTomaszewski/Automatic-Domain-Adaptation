import torch
from argparse import ArgumentParser
from utils import available_torch_device


DATA_ROOT_DIR = './data'


def parse_args() -> dict:
    parser = ArgumentParser()
    
    # Data
    parser.add_argument("--train_dataset", type=str, default="mvtec",
                        choices=["mvtec"],
                        help="The name of the dataset to use (default: mvtec)")
    parser.add_argument("--val_dataset", type=str, default="visa",
                        choices=["visa"],
                        help="The name of the dataset to use (default: visa)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="The number of classes in the dataset (default: 2)")
    
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
    parser.add_argument("--epochs", type=int, default=20,
                        help="The number of epochs to train (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The batch size for training (default: 32)")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="The batch size for evaluation (default: 32)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help="The learning rate for training (default: 1e-3)")
    parser.add_argument("--img_size", type=tuple, default=(224, 224),
                        help="The size of the images (default: (224, 224))")
    
    return parser.parse_args()
