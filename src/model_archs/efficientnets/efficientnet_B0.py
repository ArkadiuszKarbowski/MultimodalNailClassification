import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1

        # Create backbones
        def create_backbone():
            model = models.efficientnet_b0(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            # Fine-tune final blocks
            for param in model.features[-4:].parameters():
                param.requires_grad = True
            # Remove classifier
            return nn.Sequential(*list(model.children())[:-1])

        self.normal_branch = create_backbone()
        self.uv_branch = create_backbone()

        # EfficientNet-B0 has 1280 features
        feature_dim = 1280

        # Cross-attention components
        self.cross_attention = nn.ModuleDict(
            {
                "normal": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 2, feature_dim),
                    nn.Sigmoid(),
                ),
                "uv": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 2, feature_dim),
                    nn.Sigmoid(),
                ),
            }
        )

        # Classifier with doubled feature dimensions
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x_normal, x_uv):
        # Extract features
        f_normal = torch.flatten(self.normal_branch(x_normal), 1)
        f_uv = torch.flatten(self.uv_branch(x_uv), 1)

        # Cross-attention mechanism
        uv_attention = self.cross_attention["normal"](f_uv)
        attended_normal = f_normal * uv_attention

        normal_attention = self.cross_attention["uv"](f_normal)
        attended_uv = f_uv * normal_attention

        # Feature fusion
        fused = torch.cat([attended_normal, attended_uv], dim=1)

        return self.classifier(fused)
