import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from transformers import Trainer, TrainingArguments

from logger import logger
from config import parse_args
from models import DINOv2Classifier
from utils.metrics import compute_metrics
from data_loading.datasets import get_dataset
from data_loading.processing import create_data_transformation_pipeline


def create_training_args(args):   
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=args.device != torch.device('cuda'),
        warmup_steps=500
    )


if __name__ == '__main__':
    args = parse_args()
    training_args = create_training_args(args)

    logger.info(f"Training arguments: {training_args}")

    # Transformation pipeline
    transforms = create_data_transformation_pipeline(args.img_size)
    
    # Dataset
    train_dataset = get_dataset(args.train_dataset,
                                transforms=transforms,
                                is_test=False)
    val_dataset = get_dataset(args.val_dataset,
                              transforms=transforms,
                              is_test=True)

    # Model
    model = DINOv2Classifier(args.num_classes, backbone=args.dinov2_backbone, layers=1, device=args.device).to(args.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    lr_scheduler = PolynomialLR(optimizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=compute_metrics
    )

    trainer.train()
