import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models import DINOv2Classifier
from config import parse_args
from logger import logger
from data_loading.datasets import get_dataset
from data_loading.processing import create_data_transformation_pipeline


def train_one_epoch(model: nn.Module, 
                    dataloader: DataLoader, 
                    optimizer: Optimizer, 
                    loss: callable, 
                    device):
    model.train()
    for idx, data in enumerate(dataloader):
        img = data["img"].to(device)
        label = data["label"].to(device)
        
        output = model(img)
        loss_value = loss(output, label)
        
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        
        logger.info(f"Batch: {idx}, Loss: {loss_value.item()}")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    loss_history = []
    
    model.train()
    for i, data in tqdm(enumerate(dataloader)):
        X, y = data["img"].to(device), data["label"].to(device)

        outputs = model(X)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if i % 100 == 0:
            logger.info(f"Batch: {i}, Loss: {loss.item()}")
    logger.info(f"Loss after epoch {i}: {sum(loss_history) / len(loss_history)}")


def evaluate(model, dataloader, loss_fn, device):
    loss_history = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            X, y = data["img"].to(device), data["label"].to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss_history.append(loss.item())

    logger.info(f"Validation Loss: {sum(loss_history) / len(loss_history)}")


if __name__ == '__main__':
    args = parse_args()
    
    logger.info(f"Arguments: {args}")
    
    # Transformation pipeline
    transforms = create_data_transformation_pipeline(args.img_size)
    
    # Dataset
    train_dataset = get_dataset(args.train_dataset,
                                transforms=transforms,
                                is_test=False)
    val_dataset = get_dataset(args.val_dataset,
                              transforms=transforms,
                              is_test=True)

    # Dataloader
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    # Model
    model = DINOv2Classifier(args.num_classes, backbone=args.dinov2_backbone, layers=1, device=args.device).to(args.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    loss = CrossEntropyLoss()

    # training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch: {epoch}")
        train_one_epoch(model, train_dataloader, optimizer, loss, args.device)
        evaluate(model, val_dataloader, loss, args.device)
