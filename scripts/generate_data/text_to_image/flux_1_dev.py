import os
import torch

from diffusers import FluxPipeline
from dotenv import load_dotenv

load_dotenv()


model_name = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
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

    images = pipe(
        prompts,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images
    
    for prompt, image in zip(prompts, images):
        # Save the image
        image.save(os.path.join(outputs_dir, f"{prompt}.png"))
