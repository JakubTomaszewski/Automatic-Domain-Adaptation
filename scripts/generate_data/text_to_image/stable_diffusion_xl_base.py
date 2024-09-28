import os
import torch

from diffusers import DiffusionPipeline
from dotenv import load_dotenv

load_dotenv()


model_name = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

prompts = [
    "Faulty screw",
    "Head CT scan of a patient with a brain tumor",
    "Printed circuit board with faulty connections",
    "Satellite image of terrain after flooding",
]


if __name__ == '__main__':
    outputs_dir = os.path.join("results", model_name.split("/")[-1])
    os.makedirs(outputs_dir, exist_ok=True)

    images = pipe(prompt=prompts).images
    
    for prompt, image in zip(prompts, images):
        image.save(os.path.join(outputs_dir, f"{prompt}.png"))
