import pandas as pd
import numpy as np

df = pd.read_parquet('data/labels/training_labels_clean.parquet')

print('zone_age_bars stats:')
print(df['zone_age_bars'].describe())
print()

# Check if zone_age_bars correlates suspiciously with label
df['win'] = (df['label'] >= 1.0).astype(int)
bins = [0, 10, 50, 200, 1000, 99999]
labels = ['0-10', '10-50', '50-200', '200-1000', '1000+']
df['age_bucket'] = pd.cut(df['zone_age_bars'], bins=bins, labels=labels)
print('Win rate by zone_age_bars bucket:')
print(df.groupby('age_bucket', observed=True)['win'].agg(['mean','count']).round(3))
