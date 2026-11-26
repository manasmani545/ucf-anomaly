import torch
import torch.nn as nn


class GRUAttentionMIL(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=256, k_ratio=0.25):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.k_ratio = k_ratio

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, mask):
        projected = self.projection(features)

        gru_output, _ = self.gru(projected)

        attention_scores = self.attention(gru_output).squeeze(-1)

        masked_scores = attention_scores.masked_fill(~mask, -10000.0)

        attention_weights = torch.softmax(masked_scores, dim=1)

        context = torch.sum(
            gru_output * attention_weights.unsqueeze(-1),
            dim=1
        )

        logits = self.classifier(context).squeeze(-1)

        return logits
