import torch
import torch.nn as nn


class MeanPoolingBaseline(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features, mask):
        mask = mask.unsqueeze(-1).float()

        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(pooled)

        return logits, None
