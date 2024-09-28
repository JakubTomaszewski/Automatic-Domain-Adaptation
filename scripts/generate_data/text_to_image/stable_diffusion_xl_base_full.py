import os
import torch

from diffusers import DiffusionPipeline
from dotenv import load_dotenv

load_dotenv()


model_name = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")





from diffusers import DiffusionPipeline
import torch

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]





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
        # Save the image
        image.save(os.path.join(outputs_dir, f"{prompt}.png"))
