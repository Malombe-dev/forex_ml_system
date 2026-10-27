import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df = pd.read_parquet("data/labels/training_labels_clean.parquet")
d = df[np.isclose(df.label,0)|np.isclose(df.label,1)].copy()
d["zone_time"] = pd.to_datetime(d.zone_time)
t = d[d.zone_time >= "2024-02-01"]
auc = roc_auc_score(t.label, -t.approach_distance)
print("approach_distance alone, AUC:", round(auc, 3))
print(t.groupby(pd.cut(t.approach_distance,[-1,2,5,10,20,50]), observed=True).label.agg(["mean","size"]))
print()
print("by tf_source:")
print(t.groupby("tf_source").approach_distance.describe()[["mean","std","min","max"]])
