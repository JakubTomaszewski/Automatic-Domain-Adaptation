import os
import torch

from ..base_generation_model import BaseGenerationModel
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image


class StableDiffusion3(BaseGenerationModel):
    def __init__(self, model_name="stabilityai/stable-diffusion-3-medium-diffusers"):
        self.model_name = model_name
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.pipeline.to("cuda")
        self.pipeline.enable_model_cpu_offload()

    def generate_images(
        self,
        prompts,
        images,
        output_dir="generated_samples",
    ) -> list[str]:
        output_paths = []

        for i, (prompt, img_path) in enumerate(zip(prompts, images)):
            print("Generating images for prompt:", prompt)

            init_image = load_image(img_path).convert("RGB").resize((512, 512))
            image = self.pipeline(prompt=prompt, image=init_image, strength=0.8).images[0]
            # TODO: output_class_dir
            output_path = os.path.join(output_dir, f"{prompt}_{i}.png")
            image.save(output_path)
            output_paths.append(output_path)
        return output_paths
