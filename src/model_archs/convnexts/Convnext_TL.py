import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


class Model(nn.Module):
    def __init__(self, num_classes, initial_freeze=True):
        super().__init__()
        self.num_classes = num_classes
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        self.num_features_per_branch = 768  # Output features of ConvNeXt-Tiny
        input_features_classifier = self.num_features_per_branch * 2
        hidden_features = 512
        dropout_rate = 0.5

        # Normal images branch
        base_model_normal = models.convnext_tiny(weights=weights)
        self.normal_branch = nn.Sequential(*base_model_normal.features)
        self.normal_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # UV images branch
        base_model_uv = models.convnext_tiny(weights=weights)
        self.uv_branch = nn.Sequential(*base_model_uv.features)
        self.uv_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initial backbone freezing
        if initial_freeze:
            for branch in [self.normal_branch, self.uv_branch]:
                for param in branch.parameters():
                    param.requires_grad = False

        # Classification
        self.classifier = nn.Sequential(
            nn.GroupNorm(32, input_features_classifier),
            nn.Linear(input_features_classifier, hidden_features),
            nn.ReLU(),
            nn.GroupNorm(32, hidden_features),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, num_classes),
        )

    def unfreeze_last_block(self):
        print("Unfreezing the last block in both ConvNext-Tiny branches...")
        for branch in [self.normal_branch, self.uv_branch]:
            # Assuming the last block is the last few layers in the features sequence
            if len(branch) > 0:
                # Unfreeze the last block (e.g., the last 2 layers)
                for param in branch[-2:].parameters():
                    param.requires_grad = True
                print(
                    "  - Unfrozen parameters in the last block of ConvNext-Tiny branch"
                )
            else:
                print(
                    "  - Warning: Could not identify last block in branch for unfreezing."
                )

    def forward(self, x_normal, x_uv):
        f_normal = self.normal_branch(x_normal)
        f_normal = self.normal_avgpool(f_normal)  # Apply adaptive average pooling
        f_normal = torch.flatten(f_normal, 1)

        f_uv = self.uv_branch(x_uv)
        f_uv = self.uv_avgpool(f_uv)  # Apply adaptive average pooling
        f_uv = torch.flatten(f_uv, 1)

        merged_features = torch.cat([f_normal, f_uv], dim=1)

        output = self.classifier(merged_features)
        return output
