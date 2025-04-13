import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

class Model(nn.Module):
    
    def __init__(self, num_classes, initial_freeze=True):
        super().__init__()
        self.num_classes = num_classes
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        
        # MobileNetV3-Small has 576 output features
        self.num_features_per_branch = 576
        input_features_classifier = self.num_features_per_branch * 2
        hidden_features = 512
        dropout_rate = 0.5

        # Normal images branch
        base_model_normal = models.mobilenet_v3_small(weights=weights)
        self.normal_branch = nn.Sequential(*list(base_model_normal.children())[:-1]) 
        
        # UV images branch
        base_model_uv = models.mobilenet_v3_small(weights=weights)
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
        print("Unfreezing the last block in both MobileNetV3 branches...")
        for branch in [self.normal_branch, self.uv_branch]:
            # For MobileNetV3, we're looking to unfreeze the last InvertedResidual block
            if hasattr(branch, 'features') and len(branch.features) > 0:
                # Typically last couple layers in the features sequence
                for param in branch.features[-2:].parameters():
                    param.requires_grad = True
                print(f"  - Unfrozen parameters in the last block of MobileNetV3 branch")
            else:
                print(f"  - Warning: Could not identify last block in branch for unfreezing.")

    def forward(self, x_normal, x_uv):
        # Process normal images
        f_normal = self.normal_branch(x_normal)
        f_normal = torch.flatten(f_normal, 1)
        
        # Process UV images
        f_uv = self.uv_branch(x_uv)
        f_uv = torch.flatten(f_uv, 1)
        
        # Merge features
        merged_features = torch.cat([f_normal, f_uv], dim=1)
        
        # Classification
        output = self.classifier(merged_features)
        return output