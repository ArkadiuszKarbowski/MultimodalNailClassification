import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Model(nn.Module):
    
    def __init__(self, num_classes, initial_freeze=True):
        super().__init__()
        self.num_classes = num_classes
        weights = ResNet50_Weights.IMAGENET1K_V2
        
        self.num_features_per_branch = 2048
        input_features_classifier = self.num_features_per_branch * 2
        hidden_features = 512
        dropout_rate = 0.5

        # Normal images branch
        base_model_normal = models.resnet50(weights=weights)
        self.normal_branch = nn.Sequential(*list(base_model_normal.children())[:-1])

        # UV images branch
        base_model_uv = models.resnet50(weights=weights)
        self.uv_branch = nn.Sequential(*list(base_model_uv.children())[:-1])

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
            nn.Linear(hidden_features, num_classes)
        )
    def unfreeze_last_block(self):
        print("Unfreezing the last block (layer4) in both ResNet50 branches...")
        for branch in [self.normal_branch, self.uv_branch]:
            # ResNet50 modules: [0]conv1, [1]bn1, [2]relu, [3]maxpool,
            # [4]layer1, [5]layer2, [6]layer3, [7]layer4
            if len(branch) > 7 and isinstance(branch[-2], nn.Sequential):
                for param in branch[-2].parameters():
                    param.requires_grad = True
                print(f"  - Unfrozen parameters in {branch[-2].__class__.__name__} (layer4?)")
            else:
                print(f"  - Warning: Could not identify last block in branch for unfreezing.")

    def forward(self, x_normal, x_uv):
        f_normal = self.normal_branch(x_normal)
        f_uv = self.uv_branch(x_uv)

        f_normal = torch.flatten(f_normal, 1)
        f_uv = torch.flatten(f_uv, 1)

        merged_features = torch.cat([f_normal, f_uv], dim=1)
        
        output = self.classifier(merged_features)
        return output
