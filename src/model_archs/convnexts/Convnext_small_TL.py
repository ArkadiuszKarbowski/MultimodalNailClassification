import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights

class Model(nn.Module):
    
    def __init__(self, num_classes, initial_freeze=True):
        super().__init__()
        self.num_classes = num_classes
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        
        # ConvNeXt-Small has 768 output features
        self.num_features_per_branch = 768
        input_features_classifier = self.num_features_per_branch * 2
        hidden_features = 512
        dropout_rate = 0.6  
        
        # Normal images branch
        base_model_normal = models.convnext_small(weights=weights)
        self.normal_branch = nn.Sequential(*base_model_normal.features)
        self.normal_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # UV images branch
        base_model_uv = models.convnext_small(weights=weights)
        self.uv_branch = nn.Sequential(*base_model_uv.features)
        self.uv_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initial backbone freezing
        if initial_freeze:
            for branch in [self.normal_branch, self.uv_branch]:
                for param in branch.parameters():
                    param.requires_grad = False

        # classifier 
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_features_classifier),
            nn.Dropout(0.3),  # First dropout layer
            nn.Linear(input_features_classifier, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features),
            nn.Dropout(dropout_rate),  # Second dropout layer
            nn.Linear(hidden_features, num_classes)
        )
        
        # Initialize the final linear layer with smaller weights
        nn.init.xavier_normal_(self.classifier[-1].weight, gain=0.5)
    
    def unfreeze_last_block(self):
        print("Unfreezing the last block in both ConvNeXt-Small branches...")
        for branch in [self.normal_branch, self.uv_branch]:
            # Only unfreeze the very last block to minimize overfitting
            if len(branch) > 0:
                for param in branch[-1:].parameters():
                    param.requires_grad = True
                print(f"  - Unfrozen parameters in the last block of ConvNeXt-Small branch")
            else:
                print(f"  - Warning: Could not identify last block in branch for unfreezing.")

    def forward(self, x_normal, x_uv):
        # Process normal images
        f_normal = self.normal_branch(x_normal)
        f_normal = self.normal_avgpool(f_normal)
        f_normal = torch.flatten(f_normal, 1)
        
        # Process UV images
        f_uv = self.uv_branch(x_uv)
        f_uv = self.uv_avgpool(f_uv)
        f_uv = torch.flatten(f_uv, 1)
        
        # Merge features
        merged_features = torch.cat([f_normal, f_uv], dim=1)
        
        # Classification
        output = self.classifier(merged_features)
        return output