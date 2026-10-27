"""Advice per signal from frozen rules.

  python advisor.py --features features.csv --rules rules_v0.json --out advice.csv

Pre-entry points (0-5) are computed from bars complete BEFORE the touch, so they can gate the order.
The wick flag exists only after the touch bar closes: it is an extra conviction note, never an exit signal
(tested: cutting low-wick trades early lowered total R at every threshold).
"""
import argparse, json, operator
import pandas as pd, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--rules',default='rules_v0.json'); ap.add_argument('--out',default='advice.csv')
a=ap.parse_args()
OPS={'>=':operator.ge,'>':operator.gt,'<=':operator.le,'<':operator.lt}
cfg=json.load(open(a.rules)); df=pd.read_csv(a.features)
pts=np.zeros(len(df),dtype=int); reasons=[[] for _ in range(len(df))]
for r in cfg['pre_entry_rules']:
    ok=OPS[r['op']](df[r['feature']],r['value']).fillna(False).values
    pts+=ok.astype(int)
    for i in np.where(~ok)[0]: reasons[i].append(f"{r['feature']} {r['op']} {r['value']} failed ({df[r['feature']].iloc[i]:.2f})")
df['points']=pts
t=cfg['tiers']
df['advice']=np.where(pts>=t['TAKE_HIGH'],'TAKE - high conviction',np.where(pts>=t['TAKE'],'TAKE','SKIP / low conviction'))
df['failed_checks']=['; '.join(x) for x in reasons]
w=cfg.get('post_touch_rule')
if w and w['feature'] in df:
    strong=OPS[w['op']](df[w['feature']],w['value']).fillna(False)
    df['touch_note']=np.where(strong,'strong rejection: keep / consider size-up or stage-3 adjust','no strong rejection: HOLD to SL/TP (do not exit early)')
keep=[c for c in ['signal_id','symbol','direction','trigger_time','points','advice','touch_note','failed_checks','outcome','replay_R'] if c in df]
df[keep].to_csv(a.out,index=False)
print(df.advice.value_counts().to_string())
if 'replay_R' in df:
    g=df.groupby('advice').agg(signals=('points','size'),win_rate=('replay_R',lambda s:(s.dropna()>0).mean()),avg_R=('replay_R','mean'))
    print('\n(report outcomes, as recorded by the replay)'); print(g.round(2).to_string())
print('wrote',a.out)
