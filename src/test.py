from models import DINOv2Classifier
from config import parse_args

from logger import logger
from torch.utils.data import DataLoader
from data_loading.datasets import get_dataset
from data_loading.processing import create_data_transformation_pipeline


if __name__ == '__main__':
    args = parse_args()
    
    logger.info(f"Arguments: {args}")
    
    # Transformation pipeline
    transforms = create_data_transformation_pipeline(args.img_size)
    
    # Dataset
    dataset = get_dataset(args.dataset_name,
                          transforms=transforms,
                          is_test=True,
                          meta_file=args.meta_file)
    
    # Dataloader
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
    
    # Model
    model = DINOv2Classifier(dataset.num_classes, backbone=args.dinov2_backbone, layers=1, device=args.device).to(args.device)

    model.eval()
    for idx, data in enumerate(dataloader):
        img = data["img"]
        label = data["label"]
        
        logger.info(f"Img dtype: {img.dtype}, Img shape: {img.shape}, Label: {label}")
        logger.info(f"Model dtype: {model.type}")
        
        output = model(img)
        
        logger.info(f"Batch: {idx}, Output: {output}, Label: {label}")
        break
