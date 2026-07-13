# production_predictor.py — Modularization Plan

> [!NOTE]
> **Status: Conceptual Plan / Archive**
> Currently, the live trading system uses a unified monolithic file (`production_predictor.py`) to prevent deployment overhead, simplify dependencies, and keep execution self-contained. 
> 
> If you decide to modularize the codebase in the future, use this map as a guide. Note that line number ranges refer to an older revision of `production_predictor.py` and should be verified before splitting.

---

## Proposed Split Module Map

| File | What lives here | Original File Sections |
| :--- | :--- | :--- |
| `api_client.py` | `_RateLimiter`, `fetch_bars()`, `calculate_indicators()`, shared bar cache | Core API client and rate limiting logic |
| `htf_analyzer.py` | `HTFInstitutionalAnalyzer` — FVG, OB, market structure, Fibonacci, confluence scoring | HTF analysis features |
| `zone_tracker.py` | `ActiveZoneTracker` — zone deduplication, TTL/distance expiry, JSON persistence | Active zone management state |
| `signal_store.py` | `save_signals_to_single_file()`, `_convert_numpy()` | Signal JSON export helpers |
| `strategy_concepts.py` | Zone detection (`_detect_zones_with_validation`), direction filter, zone flip (CONTINUATION), 5-min refinement, conflict suppressor | Strategy entry conditions & zones |
| `htf_confluence.py` | `_check_1h_confluence()`, `_check_5min_entry()`, `_get_15min_approach()` | Confluence checklist verification |
| `schedulers.py` | `DailyPlanScheduler`, `WeeklyPlanScheduler`, `PriceZoneMonitor` | Scheduling and execution triggers |
| `main.py` | `run_dual_strategy()`, predictor factory, main loop | Live runner entrypoint and scheduler loop |
| `predictor.py` | `ProductionForexPredictor` class wrapper importing and stitching all the above modules together | Core class logic |

---

## Modularization wiring guidelines

When wiring the classes, the `ProductionForexPredictor` class in `predictor.py` would import and attach these helper functions as instance methods using a mixin pattern:

```python
# predictor.py
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from api_client import fetch_bars, calculate_indicators
from htf_analyzer import HTFInstitutionalAnalyzer
from zone_tracker import ActiveZoneTracker
from signal_store import save_signals_to_single_file

# Import standalone functions to bind as instance methods
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
        # Initialization logic
        ...
```

---

## Import Graph (Dependency Resolution)

```text
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
