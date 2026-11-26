import pandas as pd
from pathlib import Path
df = pd.read_csv('data/processed/manifest.csv')
df['feature_path'] = df['feature_path'].str.replace('\\', '/', regex=False).str.split('hf_download/').str[-1]
df['feature_dim'] = 2048
Path('kaggle').mkdir(exist_ok=True)
df.to_csv('kaggle/manifest_kaggle.csv', index=False)
print(df.head(2).to_string())
