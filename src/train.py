import torch

from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

from logger import logger
from config import parse_args
from models import DINOv2Classifier
from utils.metrics import compute_metrics
from data_loading.datasets import get_dataset
from data_loading.processing import create_data_transformation_pipeline

from dotenv import load_dotenv

load_dotenv()


def create_training_args(args):   
    return TrainingArguments(
        output_dir=args.output_dir,
        report_to="wandb",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        no_cuda=args.device != torch.device('cuda'),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
    )


if __name__ == '__main__':
    args = parse_args()
    training_args = create_training_args(args)

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Training dataset: {args.train_dataset}")
    logger.info(f"Validation dataset: {args.val_dataset}")

    # Transformation pipeline
    transforms = create_data_transformation_pipeline(args.img_size)
    
    # Dataset
    train_dataset = get_dataset(args.train_dataset,
                                transforms=transforms,
                                is_test=False)
    val_dataset = get_dataset(args.val_dataset,
                              transforms=transforms,
                              is_test=True)

    class_weights = train_dataset.get_class_weights() if args.weight_classes else None

    # Model
    model = DINOv2Classifier(num_classes=args.num_classes,
                             backbone=args.dinov2_backbone,
                             class_weights=class_weights,
                             num_layers=4, 
                             device=args.device
                             ).to(args.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics
    )

    trainer.train()
