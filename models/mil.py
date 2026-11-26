import torch
import torch.nn as nn


class TopKMIL(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, k_ratio: float = 0.125):
        super().__init__()

        self.classifier = nn.Linear(feature_dim, num_classes)
        self.k_ratio = k_ratio

    def forward(self, features, mask):
        logits = self.classifier(features)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)

        batch_size = logits.shape[0]
        video_logits = []

        for i in range(batch_size):
            valid_length = int(mask[i].sum().item())
            k = max(1, int(self.k_ratio * valid_length))

            topk_logits, _ = logits[i].topk(k, dim=0)
            video_logits.append(topk_logits.mean(dim=0))

        video_logits = torch.stack(video_logits, dim=0)

        return video_logits, logits
