import torch
import torch.nn as nn


class TopKMIL(nn.Module):
    def __init__(self, feature_dim, num_classes, k_ratio=0.25):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.k_ratio = k_ratio

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

    def forward(self, features, mask):
        logits = self.encoder(features).squeeze(-1)

        batch_size, max_len = logits.shape

        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)

        valid_counts = mask.sum(dim=1)

        k_per_video = torch.clamp(
            (valid_counts.float() * self.k_ratio).ceil().long(),
            min=1
        )

        positions = torch.arange(max_len, device=logits.device).unsqueeze(0)

        selection_mask = positions < k_per_video.unsqueeze(1)

        selected_logits = sorted_logits * selection_mask.float()

        selected_count = k_per_video.float().unsqueeze(1)

        video_logits = selected_logits.sum(dim=1) / selected_count.squeeze(1)

        return video_logits
