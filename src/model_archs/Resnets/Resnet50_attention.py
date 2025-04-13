import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1

        # Feature dimension from ResNet50
        feature_dim = 2048

        # Create backbones
        def create_backbone():
            model = models.resnet50(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            for param in [
                *model.layer2.parameters(),
                *model.layer3.parameters(),
                *model.layer4.parameters(),
            ]:
                param.requires_grad = True
            return nn.Sequential(*list(model.children())[:-1])

        self.normal_branch = create_backbone()
        self.uv_branch = create_backbone()

        # Cross-modal attention components - updated dimensions
        self.cross_attention = nn.ModuleDict(
            {
                "normal": nn.Sequential(
                    nn.Linear(feature_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, feature_dim),
                    nn.Sigmoid(),
                ),
                "uv": nn.Sequential(
                    nn.Linear(feature_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, feature_dim),
                    nn.Sigmoid(),
                ),
            }
        )

        # Lightweight self-attention - updated dimensions
        self.self_attention = nn.ModuleDict(
            {
                "normal": nn.Linear(feature_dim, feature_dim),
                "uv": nn.Linear(feature_dim, feature_dim),
            }
        )

        # Modality gating mechanism - updated dimensions
        self.gate = nn.Sequential(nn.Linear(feature_dim * 2, 2), nn.Softmax(dim=1))

        # Simplified classifier - updated dimensions
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x_normal, x_uv):
        # Extract features
        f_normal = self.normal_branch(x_normal).flatten(1)  # (B, 2048)
        f_uv = self.uv_branch(x_uv).flatten(1)  # (B, 2048)

        # Apply lightweight self-attention
        normal_attn = torch.sigmoid(self.self_attention["normal"](f_normal))
        uv_attn = torch.sigmoid(self.self_attention["uv"](f_uv))

        f_normal = f_normal * normal_attn
        f_uv = f_uv * uv_attn

        # Cross-attention mechanism
        uv_attention = self.cross_attention["normal"](f_uv)
        attended_normal = f_normal * uv_attention

        normal_attention = self.cross_attention["uv"](f_normal)
        attended_uv = f_uv * normal_attention

        # Residual connections
        attended_normal = attended_normal + f_normal
        attended_uv = attended_uv + f_uv

        # Feature fusion
        fused = torch.cat([attended_normal, attended_uv], dim=1)

        # Dynamic modality weighting
        gate_weights = self.gate(fused)
        weighted_normal = attended_normal * gate_weights[:, 0].unsqueeze(1)
        weighted_uv = attended_uv * gate_weights[:, 1].unsqueeze(1)

        # Final fusion
        gated_fusion = torch.cat([weighted_normal, weighted_uv], dim=1)

        return self.classifier(gated_fusion)
