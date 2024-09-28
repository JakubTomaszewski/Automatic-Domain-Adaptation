import os
import torch

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

from dotenv import load_dotenv

load_dotenv()


model_name = "stabilityai/stable-diffusion-xl-refiner-1.0"

pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
pipeline.to("cuda")


img_prompts = [
    ("Daisy flower", "/home/jtomaszewski/Automatic-Domain-Adaptation/data/flowers/train/daisy/2488902131_3417698611_n.jpg"),
    ("Rose", "/home/jtomaszewski/Automatic-Domain-Adaptation/data/flowers/train/rose/3179751458_9646d839f6_n.jpg"),
    ("A field of roses", "/home/jtomaszewski/Automatic-Domain-Adaptation/data/flowers/train/rose/3179751458_9646d839f6_n.jpg"),
    ("Sunflower", "/home/jtomaszewski/Automatic-Domain-Adaptation/data/flowers/train/sunflower/4746638094_f5336788a0_n.jpg"),
    ("Tulip", "/home/jtomaszewski/Automatic-Domain-Adaptation/data/flowers/train/tulip/4521037085_70d5802e1d_m.jpg"),
    ("A different tulip", "/home/jtomaszewski/Automatic-Domain-Adaptation/data/flowers/train/tulip/4521037085_70d5802e1d_m.jpg")
]


if __name__ == '__main__':
    outputs_dir = os.path.join("results", model_name.split("/")[-1] + "_strength_0.8")
    os.makedirs(outputs_dir, exist_ok=True)

    for prompt, img_path in img_prompts:
        print("Generating images for prompt:", prompt)
        
        init_image = load_image(img_path)
        images = pipeline(prompt=prompt, image=init_image, strength=0.8).images

        for i, image in enumerate(images):
            # make a grid of images and save it
            grid = make_image_grid([init_image, image], rows=1, cols=2)
            grid.save(os.path.join(outputs_dir, f"{prompt}_{i}.png"))
