import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, manifest_path, split, num_classes, max_len=750):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.num_classes = num_classes
        self.max_len = max_len

        with open('data/processed/video_labels.json', 'r') as f:
            self.video_labels = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        video_id = row['video_id']

        features = np.load(row['feature_path']).astype(np.float32)

        if features.ndim == 3:
            features = features.mean(axis=1)

        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0

        features = (features - mean) / std

        features = torch.from_numpy(features)

        original_len = features.shape[0]

        if original_len > self.max_len:
            start = random.randint(0, original_len - self.max_len)
            features = features[start:start + self.max_len]
            original_len = self.max_len
        else:
            padding = self.max_len - original_len
            features = torch.nn.functional.pad(
                features,
                (0, 0, 0, padding)
            )

        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:original_len] = True

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[int(self.video_labels[video_id][0])] = 1.0

        return features, target, mask, video_id
