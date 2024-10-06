from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor
)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def create_data_augmentation_pipeline(img_size: tuple[int, int] = (224, 224)):
    return Compose(
        [
            RandomResizedCrop(img_size, ratio=(0.85, 1.0)),
            RandomHorizontalFlip(p=0.5),
        ]
    )


def create_data_transformation_pipeline(img_size: tuple[int, int] = (224, 224)):
    return Compose(
        [
            ToTensor(),
            # ConvertImageDtype(torch.float32),
            # GrayscaleToRGB(),
            Resize(img_size),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
