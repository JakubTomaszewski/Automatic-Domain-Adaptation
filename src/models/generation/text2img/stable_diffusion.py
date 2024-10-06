import os
import torch

from ..base_generation_model import BaseGenerationModel
from diffusers import StableDiffusion3Pipeline


class StableDiffusion3(BaseGenerationModel):
    def __init__(self, model_name="stabilityai/stable-diffusion-3-medium-diffusers"):
        self.model_name = model_name
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.pipeline = self.pipeline.to("cuda")

    def generate_images(
        self, prompts, output_dir="results", num_inference_steps=28, guidance_scale=7.0
    ):
        outputs_dir = os.path.join(output_dir, self.model_name.split("/")[-1])
        os.makedirs(outputs_dir, exist_ok=True)

        images = self.pipeline(
            prompts,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images

        for prompt, image in zip(prompts, images):
            image.save(os.path.join(outputs_dir, f"{prompt}.png"))


if __name__ == "__main__":
    prompts = ["a cat", "a dog"]  # Example prompts
    sd3 = StableDiffusion3()
    sd3.generate_images(prompts)
