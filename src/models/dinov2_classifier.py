import torch
from torch import nn


class DINOv2Classifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 backbone: str = "dinov2_vits14", 
                 layers: int = 1,
                 device: torch.device = torch.device('cuda'),
                 *args, 
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.layers = layers
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True).to(device)
        self.classification_head = nn.Linear((1 + layers) * self.backbone.embed_dim, num_classes)
        # TODO: check more linear layers
        # self.classifier = nn.Sequential(
        #     nn.Linear(384, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.classification_head(linear_input)
    
    def predict(self, x):
        return self.forward(x).argmax(dim=-1)
