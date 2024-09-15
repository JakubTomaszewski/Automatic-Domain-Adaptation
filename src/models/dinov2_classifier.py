import torch
from torch import nn


class DINOv2Classifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 backbone: str = "dinov2_vits14", 
                 class_weights: torch.Tensor = None,
                 num_layers: int = 1,
                 device: torch.device = torch.device('cuda'),
                 *args, 
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_layers = num_layers
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True).to(device)
        self.classification_head = nn.Sequential(
            nn.Linear((1 + self.num_layers) * self.backbone.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, labels=None):
        if self.num_layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
        elif self.num_layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
        else:
            raise ValueError(f"Unsupported number of layers: {self.num_layers}")
        logits = self.classification_head(linear_input)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}
    
    def predict(self, x):
        return self.forward(x)["logits"].argmax(dim=-1)
