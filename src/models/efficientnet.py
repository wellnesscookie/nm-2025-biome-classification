from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 with a new `num_classes`-way head."""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # --- fine-tuning helpers -----------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze everything except the final classifier."""
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> EfficientNetB0Classifier:
    """Convenience factory."""
    return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)


# Suggested input size for B0 — matches its ImageNet pretraining.
EFFICIENTNET_B0_INPUT: tuple[int, int] = (224, 224)
