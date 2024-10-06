from utils import GenerationType
from .text2img.stable_diffusion import StableDiffusion3 as StableDiffusion3Text2Img
from .img2img.stable_diffusion import StableDiffusion3 as StableDiffusion3Img2Img
from .img2img.unclip import unCLIP


TEXT_TO_IMG_MODELS = {
    "StableDiffusion3Text2Img": StableDiffusion3Text2Img,
}

IMG_TO_IMG_MODELS = {
    "StableDiffusion3Img2Img": StableDiffusion3Img2Img,
    "unCLIP": unCLIP,
}


def get_model(model_name, generation_type: GenerationType):
    if generation_type == GenerationType.TEXT_TO_IMG:
        return TEXT_TO_IMG_MODELS[model_name]()
    elif generation_type == GenerationType.IMG_TO_IMG:
        return IMG_TO_IMG_MODELS[model_name]()
    else:
        raise ValueError(f"Invalid generation type: {generation_type}")
