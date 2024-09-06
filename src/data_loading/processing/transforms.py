import torch


class GrayscaleToRGB:
    def __call__(self, image: torch.Tensor):
        if image.size(0) == 1:
            return image.repeat(3, 1, 1)
        else:
            return image
