import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.feature_dataset import FeatureDataset
from models.mil_model import TopKMIL


def main():
    checkpoint_path = Path('checkpoints/ucf_baseline/best_account_a.pt')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    test_dataset = FeatureDataset(
        config['dataset']['manifest'],
        'test',
        config['dataset']['num_classes'],
        config['dataset']['max_video_len']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    model = TopKMIL(
        feature_dim=config['dataset']['feature_dim'],
        num_classes=config['dataset']['num_classes'],
        k_ratio=config['model']['k_ratio']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_targets = []
    all_scores = []
    all_video_ids = []

    with torch.no_grad():
        for features, targets, masks, video_ids in tqdm(test_loader, desc='Evaluating'):
            logits = model(features, masks)
            scores = torch.sigmoid(logits)

            all_targets.extend(targets[:, 1].numpy())
            all_scores.extend(scores.numpy())
            all_video_ids.extend(list(video_ids))

    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = None

    for threshold in thresholds:
        preds = (all_scores >= threshold).astype(int)

        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        f1 = f1_score(all_targets, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

            best_metrics = {
                'threshold': best_threshold,
                'accuracy': float(accuracy_score(all_targets, preds)),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(roc_auc_score(all_targets, all_scores)),
                'confusion_matrix': confusion_matrix(all_targets, preds).tolist()
            }

    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'ucf_baseline_threshold_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)

    print(json.dumps(best_metrics, indent=2))


if __name__ == '__main__':
    main()
