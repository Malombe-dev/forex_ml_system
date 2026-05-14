import pandas as pd
from pathlib import Path

bars_path = Path("data/bars_15min.parquet")
if bars_path.exists():
    df = pd.read_parquet(bars_path)
    print("15min Data Range:", df['time'].min(), "to", df['time'].max())


import pandas as pd
zones_df = pd.read_parquet("data/features/merged_zones.parquet")
# Assuming there is a time or open_time column in the zones DataFrame
print("Zones Date Range:", zones_df['time'].min(), "to", zones_df['time'].max())    