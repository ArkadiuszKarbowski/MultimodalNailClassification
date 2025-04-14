from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        def create_backbone():
            model = convnext_tiny(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            # Fine-tune the last few stages
            for param in model.features[5:].parameters():
                param.requires_grad = True
            return nn.Sequential(*list(model.children())[:-1])  # Remove classifier

        self.normal_branch = create_backbone()
        self.uv_branch = create_backbone()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cross_attention = nn.ModuleDict(
            {
                "normal": nn.Sequential(
                    nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 768), nn.Sigmoid()
                ),
                "uv": nn.Sequential(
                    nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 768), nn.Sigmoid()
                ),
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_normal, x_uv):
        f_normal = self.normal_branch(x_normal)
        f_normal = self.avgpool(f_normal).flatten(1)
        f_uv = self.uv_branch(x_uv)
        f_uv = self.avgpool(f_uv).flatten(1)

        uv_attention = self.cross_attention["normal"](f_uv)
        attended_normal = f_normal * uv_attention

        normal_attention = self.cross_attention["uv"](f_normal)
        attended_uv = f_uv * normal_attention

        fused = torch.cat([attended_normal, attended_uv], dim=1)
        return self.classifier(fused)
