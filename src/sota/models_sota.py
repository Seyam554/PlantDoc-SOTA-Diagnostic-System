import torch
import torch.nn as nn
import timm

class DINOv2Classifier(nn.Module):
    def __init__(self, model_name="dinov2_vits14", num_classes=28, freeze_backbone=False, dropout=0.3):
        super().__init__()
        self.model_name = model_name
        
        # Load DINOv2 from torch.hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Advanced multi-layer classification head with normalization and residual features
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # DINOv2 returns dict or cls token
        features = self.backbone(x)
        logits = self.head(features)
        return logits

class ConvNeXtClassifier(nn.Module):
    def __init__(self, model_name="convnext_base.fb_in22k_ft_in1k", num_classes=28, pretrained=True, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x):
        return self.model(x)

class SwinClassifier(nn.Module):
    def __init__(self, model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k", num_classes=28, pretrained=True, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x):
        return self.model(x)

def get_sota_model(arch="dinov2_vits14", num_classes=28, pretrained=True, dropout=0.3):
    arch = arch.lower()
    if "dinov2" in arch:
        return DINOv2Classifier(model_name=arch, num_classes=num_classes, dropout=dropout)
    elif "convnext" in arch:
        return ConvNeXtClassifier(model_name=arch, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif "swin" in arch:
        return SwinClassifier(model_name=arch, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
