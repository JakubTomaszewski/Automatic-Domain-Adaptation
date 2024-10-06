"""Generates samples using a generative model.

1. Loads a meta.json file
2. For each class in the meta file, generates NUM_SAMPLES_PER_CLASS - CURRENT_SAMPLES_PER_CLASS samples
3. Saves the generated samples in the appropriate directory
4. Updates the meta.json file with the new samples
"""

import os
import sys
import json
import random

sys.path.append("./src")

from argparse import ArgumentParser
from utils import GenerationType
from models.generation import get_model, TEXT_TO_IMG_MODELS, IMG_TO_IMG_MODELS

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=[*TEXT_TO_IMG_MODELS.keys(), *IMG_TO_IMG_MODELS.keys()], default="StableDiffusion3Img2Img")
    parser.add_argument("--num_samples_per_class", type=int, default=10)
    parser.add_argument("--meta_file", type=str, default="meta.json")
    parser.add_argument("--output_meta_file", type=str, default="meta_generations.json")
    parser.add_argument("--output_dir", type=str, default="generated_samples")
    parser.add_argument("--generation_type", type=GenerationType, default=GenerationType.IMG_TO_IMG)
    args = parser.parse_args()

    with open(args.meta_file, "r") as f:
        meta_data = json.load(f)

    model = get_model(args.model, args.generation_type)

    working_dir = os.path.dirname(args.meta_file)
    os.chdir(working_dir)

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving generated samples to {output_dir}")

    for class_name, class_samples in meta_data["train"].items():
        num_samples_to_generate = max(args.num_samples_per_class - len(class_samples), 0)
        print(f"Generating {num_samples_to_generate} samples for {class_name}")

        if args.generation_type == GenerationType.IMG_TO_IMG:
            # An LLM would generate a prompt here
            base_prompts = [class_name] * num_samples_to_generate
            # Randomly pick images
            base_image_paths = [sample["img_path"] for sample in random.choices(class_samples, k=num_samples_to_generate)]
            generated_samples = model.generate_images(prompts=base_prompts, images=base_image_paths, output_dir=output_dir)
            generated_samples_dict = [{"img_path": img_path,
                                       "cls_name": class_samples[0]["cls_name"],
                                       "label": class_samples[0]["label"]
                                       } for img_path in generated_samples]

        elif args.generation_type == GenerationType.TEXT_TO_IMG:
            base_prompts = [class_name] * num_samples_to_generate
            generated_samples = model.generate_images(prompts=base_prompts, output_dir=output_dir)
            generated_samples_dict = [{"img_path": img_path,
                                       "cls_name": class_name,
                                       "label": class_samples[0]["label"]
                                       } for img_path in generated_samples]
        
        # TODO: Optimize using batching

        # Update meta_data
        meta_data["train"][class_name].extend(generated_samples_dict)

    # Save the updated meta_data
    with open(args.output_meta_file, "w") as f:
        json.dump(meta_data, f)

        
