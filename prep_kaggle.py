import numpy as np
from pathlib import Path
from tqdm import tqdm

src = Path('data/raw/ucf_crime/features/hf_download')
dst = Path('data/kaggle_features')
for f in tqdm(list(src.rglob('*.npy'))):
    out = dst / f.relative_to(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    a = np.load(f)
    if a.ndim == 3:
        a = a.mean(axis=1)
    np.save(out, a.astype(np.float32))
