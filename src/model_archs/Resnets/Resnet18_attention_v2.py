import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1

        # Create backbones
        def create_backbone():
            model = models.resnet18(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            for param in [*model.layer3.parameters(), *model.layer4.parameters()]:
                param.requires_grad = True
            return nn.Sequential(*list(model.children())[:-1])

        self.normal_branch = create_backbone()
        self.uv_branch = create_backbone()

        # Cross-attention components
        self.cross_attention = nn.ModuleDict(
            {
                "normal": nn.Sequential(
                    nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 512), nn.Sigmoid()
                ),
                "uv": nn.Sequential(
                    nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 512), nn.Sigmoid()
                ),
            }
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GroupNorm(32, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),   
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_normal, x_uv):
        # Extract features
        f_normal = self.normal_branch(x_normal).flatten(1)  # (B, 512)
        f_uv = self.uv_branch(x_uv).flatten(1)  # (B, 512)

        # Cross-attention mechanism
        uv_attention = self.cross_attention["normal"](f_uv)
        attended_normal = f_normal * uv_attention

        normal_attention = self.cross_attention["uv"](f_normal)
        attended_uv = f_uv * normal_attention

        # Feature fusion
        fused = torch.cat([attended_normal, attended_uv], dim=1)

        return self.classifier(fused)
