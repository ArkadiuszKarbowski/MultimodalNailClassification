import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1

        # Feature extractors with partial unfreezing
        def create_backbone():
            model = models.resnet18(weights=weights)

            # Freeze first 3 blocks
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze last block
            for param in model.layer4.parameters():
                param.requires_grad = True

            return nn.Sequential(*list(model.children())[:-1])

        self.normal_branch = create_backbone()
        self.uv_branch = create_backbone()

        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
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

        fused = weighted_normal + weighted_uv + 0.2 * (f_normal + f_uv)

        return self.classifier(fused)
