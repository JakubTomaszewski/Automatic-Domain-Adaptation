import torch

from torchvision.transforms import Compose, Resize, ConvertImageDtype, Grayscale, Normalize
from .transforms import GrayscaleToRGB


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def create_data_transformation_pipeline(img_size: tuple[int, int] = (224, 224)):
    return Compose([
        GrayscaleToRGB(),
        Resize(img_size),
        ConvertImageDtype(torch.float32),
        # Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
