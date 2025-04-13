import torch
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        def create_backbone():
            model = convnext_tiny(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            for block in model.features[-3:]:
                for param in block.parameters():
                    param.requires_grad = True
            return nn.Sequential(*list(model.children())[:-1])

        self.normal_branch = create_backbone()
        self.uv_branch = create_backbone()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm_normal = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(0.2)
        )
        self.norm_uv = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(0.2)
        )
        
        self.cross_attention = nn.ModuleDict({
            'normal': nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Linear(768, 768),
            ),
            'uv': nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Linear(768, 768),
            )
        })
        
        self.classifier = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_in',
                    nonlinearity='relu'  # GELU is approximated as ReLU
                )
                nn.init.constant_(m.bias, 0.01)
 
        self.hidden_act = 'gelu' # for ConvNeXt compatibility

        self.cross_attention.apply(_init_weights)
        self.classifier.apply(_init_weights)

    def forward(self, x_normal, x_uv):
        f_normal = self.normal_branch(x_normal)
        f_normal = self.avgpool(f_normal)
        f_normal = self.norm_normal(f_normal)
        
        f_uv = self.uv_branch(x_uv)
        f_uv = self.avgpool(f_uv)
        f_uv = self.norm_uv(f_uv)
        
        f_normal = torch.clamp(f_normal, -10.0, 10.0)
        f_uv = torch.clamp(f_uv, -10.0, 10.0)
        
        uv_attention = torch.sigmoid(self.cross_attention['normal'](f_uv))
        attended_normal = f_normal + (f_normal * uv_attention)
        
        normal_attention = torch.sigmoid(self.cross_attention['uv'](f_normal))
        attended_uv = f_uv + (f_uv * normal_attention)
        
        fused = torch.cat([attended_normal, attended_uv], dim=1)
        fused = nn.functional.normalize(fused, p=2, dim=1)
        
        return self.classifier(fused)
