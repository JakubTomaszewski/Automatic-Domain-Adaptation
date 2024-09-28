import os
import torch

from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from dotenv import load_dotenv

load_dotenv()

prior_model_id = "kakaobrain/karlo-v1-alpha"
data_type = torch.float16
prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

prior_text_model_id = "openai/clip-vit-large-patch14"
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

model_name = "stabilityai/stable-diffusion-2-1-unclip-small"

pipe = StableUnCLIPPipeline.from_pretrained(
    model_name,
    torch_dtype=data_type,
    prior_tokenizer=prior_tokenizer,
    prior_text_encoder=prior_text_model,
    prior=prior,
    prior_scheduler=prior_scheduler,
)

pipe.enable_model_cpu_offload()
pipe.to("cuda")


prompts = [
    "Daisy flower",
    "Rose",
    "A field of roses",
    "Sunflower",
    "Tulip",
    "A different tulip", 
]


if __name__ == '__main__':
    outputs_dir = os.path.join("results", model_name.split("/")[-1])
    os.makedirs(outputs_dir, exist_ok=True)

    images = pipe(prompt=prompts).images
    
    for prompt, image in zip(prompts, images):
        image.save(os.path.join(outputs_dir, f"{prompt}.png"))
