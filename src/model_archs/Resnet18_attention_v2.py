import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MultimodalResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1

        def create_normal_backbone():
            model = models.resnet18(weights=weights)
            # Freeze first 3 blocks
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze only layer4
            for param in model.layer4.parameters():
                param.requires_grad = True
            return nn.Sequential(*list(model.children())[:-1])

        def create_uv_backbone():
            model = models.resnet18(weights=weights)
            # Freeze only early layers
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
            # Unfreeze layer2, layer3, and layer4
            for layer in [model.layer2, model.layer3, model.layer4]:
                for param in layer.parameters():
                    param.requires_grad = True
            return nn.Sequential(*list(model.children())[:-1])

        self.normal_branch = create_normal_backbone()
        self.uv_branch = create_uv_backbone()

        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),  # Increased capacity
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_normal, x_uv):
        f_normal = self.normal_branch(x_normal).flatten(1)
        f_uv = self.uv_branch(x_uv).flatten(1)

        # Attention with residual connection
        combined = torch.cat([f_normal, f_uv], dim=1)
        attention_weights = self.attention(combined)

        # Apply attention weights and add a proper residual connection
        weighted_normal = f_normal * attention_weights[:, 0:1]
        weighted_uv = f_uv * attention_weights[:, 1:2]
        self.fusion_weight = nn.Parameter(torch.tensor(0.2))
        fused = weighted_normal + weighted_uv + self.fusion_weight * (f_normal + f_uv)

        return self.classifier(fused)
