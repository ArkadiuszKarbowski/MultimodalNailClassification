import torch
import torch.nn as nn
from torchvision import models


class MultimodalResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MultimodalResNet, self).__init__()

        # Normal spectrum branch
        self.resnet_normal = models.resnet50(pretrained=pretrained)
        self.resnet_normal = nn.Sequential(
            *list(self.resnet_normal.children())[:-1]
        )  # Remove FC layer

        # UV spectrum branch
        self.resnet_uv = models.resnet50(pretrained=pretrained)
        self.resnet_uv = nn.Sequential(*list(self.resnet_uv.children())[:-1])

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512),  # 2048 features from each ResNet
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_normal, x_uv):
        # Feature extraction
        features_normal = self.resnet_normal(x_normal).flatten(1)
        features_uv = self.resnet_uv(x_uv).flatten(1)

        # Feature fusion
        combined = torch.cat((features_normal, features_uv), dim=1)

        # Classification
        return self.classifier(combined)
