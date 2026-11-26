import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path('.')
feature_dir = root / 'data/raw/ucf_crime/features/hf_download'
processed_dir = root / 'data/processed'
processed_dir.mkdir(parents=True, exist_ok=True)

train_dir = feature_dir / 'UCF_Train_ten_crop_i3d'
test_dir = feature_dir / 'UCF_Test_ten_crop_i3d'

rows = []
video_labels = {}

for split, directory in [('train', train_dir), ('test', test_dir)]:
    for path in sorted(directory.glob('*.npy')):
        video_id = path.stem
        features = np.load(path, mmap_mode='r')
        num_snippets = int(features.shape[0])
        is_anomaly = 0 if video_id.startswith('Normal_Videos') else 1

        rows.append({
            'video_id': video_id,
            'feature_path': str(path.relative_to(root)).replace('\\\\', '/'),
            'num_snippets': num_snippets,
            'feature_dim': int(features.shape[1]) if features.ndim == 2 else -1,
            'split': split,
            'label': is_anomaly
        })

        video_labels[video_id] = [is_anomaly]

df = pd.DataFrame(rows)
df.to_csv(processed_dir / 'manifest.csv', index=False)

with open(processed_dir / 'video_labels.json', 'w') as f:
    json.dump(video_labels, f, indent=2)

df[df['split'] == 'train'].to_csv(processed_dir / 'train.csv', index=False)
df[df['split'] == 'test'].to_csv(processed_dir / 'test.csv', index=False)

print(df.groupby(['split', 'label']).size())
print('Feature dimension:', df['feature_dim'].value_counts().to_dict())
