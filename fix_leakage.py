import pandas as pd

df = pd.read_parquet('data/labels/training_labels_clean.parquet')
print(f'Before: {len(df)} rows')

leaky = (df['tf_source'] == '4h') & (df['zone_sources'].astype(str).str.contains('fvg'))
print(f'Dropping {leaky.sum()} leaky 4h-FVG rows')

df_clean = df[~leaky].copy()
df_clean['h4_fvg_inside'] = 0.0

print(f'After: {len(df_clean)} rows')
print(f'Win rate after fix: {(df_clean.label >= 1.0).mean():.1%}')

df_clean.to_parquet('data/labels/training_labels_clean.parquet', compression='snappy', index=False)
print('Saved.')
