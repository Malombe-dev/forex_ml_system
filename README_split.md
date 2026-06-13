# production_predictor.py — Split Module Map

## File Overview

| File | What lives here | Replaces |
|------|-----------------|----------|
| `api_client.py` | `_RateLimiter`, `fetch_bars()`, `calculate_indicators()`, shared bar cache | Lines 241–298, 983–1109 |
| `htf_analyzer.py` | `HTFInstitutionalAnalyzer` — FVG, OB, market structure, Fibonacci, confluence scoring | Lines 302–521 |
| `zone_tracker.py` | `ActiveZoneTracker` — dedup, TTL/distance expiry, JSON persistence | Lines 525–629 |
| `signal_store.py` | `save_signals_to_single_file()`, `_convert_numpy()` | Lines 636–798 |
| `strategy_concepts.py` | Zone detection (`_detect_zones_with_validation`), direction filter, zone flip (CONTINUATION), 5min refinement, conflict suppressor | Lines 1196–1512, 2700–2712 |
| `htf_confluence.py` | `_check_1h_confluence()`, `_check_5min_entry()`, `_get_15min_approach()` | Lines 1943–2237 |
| `schedulers.py` | `DailyPlanScheduler`, `WeeklyPlanScheduler`, `PriceZoneMonitor` | Lines 3272–3555 |
| `main.py` | `run_dual_strategy()`, predictor factory, main loop | Lines 3581–3831 |
| `predictor.py` | **Still needs creating** — `ProductionForexPredictor` class that imports all the above | Lines 801–3267 |

---

## How to wire `predictor.py`

The `ProductionForexPredictor` class in the original file calls methods defined
in the modules above. The simplest approach is to import them and attach them
as instance methods using a mixin pattern:

```python
# predictor.py
import pickle, numpy as np, pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from api_client import fetch_bars, calculate_indicators
from htf_analyzer import HTFInstitutionalAnalyzer
from zone_tracker import ActiveZoneTracker
from signal_store import save_signals_to_single_file

# Import standalone functions that become bound methods
from strategy_concepts import (
    detect_zones_with_validation as _detect_zones_with_validation,
    htf_direction_filter as _htf_direction_filter,
    detect_zone_flip,
    refine_entry_with_5min,
    suppress_conflicts as _suppress_conflicts,
)
from htf_confluence import (
    check_1h_confluence as _check_1h_confluence,
    check_5min_entry as _check_5min_entry,
    get_15min_approach as _get_15min_approach,
)
from feature_builder import build_feature_row


class ProductionForexPredictor:
    # Attach imported functions as bound methods
    _detect_zones_with_validation = _detect_zones_with_validation
    _htf_direction_filter         = _htf_direction_filter
    detect_zone_flip              = detect_zone_flip
    refine_entry_with_5min        = refine_entry_with_5min
    _check_1h_confluence          = _check_1h_confluence
    _check_5min_entry             = _check_5min_entry
    _get_15min_approach           = _get_15min_approach

    def __init__(self, api_key, ...):
        ...
        # Use module-level fetch_bars, passing self.symbol and self.api_key
        # self.fetch_bars = lambda interval, outputsize: fetch_bars(
        #     self.symbol, self.api_key, interval, outputsize)
        ...
```

---

## Import graph (no circular dependencies)

```
main.py
  └── predictor.py
        ├── api_client.py          (no local imports)
        ├── htf_analyzer.py        (no local imports)
        ├── zone_tracker.py        (no local imports)
        ├── signal_store.py        (no local imports)
        ├── strategy_concepts.py   (no local imports)
        ├── htf_confluence.py      (no local imports)
        └── feature_builder.py     (existing file, unchanged)
  └── schedulers.py
        └── signal_store.py
  └── signal_store.py
```

---

## What still lives in the original file

The remaining code in `production_predictor.py` (after the split) is:
- `ProductionForexPredictor.__init__` + `load_model()`
- `get_htf_context()` — builds HTF context dict from fetched bars
- `htf_trade_detect()` — primary 4H OB detection + 1H confluence pipeline
- `predict_with_proper_features()` + `_predict()` + `_ml_score_htf_zone()` — ML scoring
- `calculate_realistic_levels()` — Entry/SL/TP calculation
- `generate_signals()` — the main signal loop (calls all of the above)

That core is ~800 lines which is reasonable for a single `predictor.py`.
