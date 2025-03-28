import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MultimodalResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1

        # Create backbones
        def create_backbone():
            model = models.resnet18(weights=weights)
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

        # Cross-modal attention components
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

        # Lightweight self-attention - only adds minimal parameters
        self.self_attention = nn.ModuleDict(
            {"normal": nn.Linear(512, 512), "uv": nn.Linear(512, 512)}
        )

        # Modality gating mechanism
        self.gate = nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=1))

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_normal, x_uv):
        # Extract features
        f_normal = self.normal_branch(x_normal).flatten(1)  # (B, 512)
        f_uv = self.uv_branch(x_uv).flatten(1)  # (B, 512)

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
