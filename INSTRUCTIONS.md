# INSTRUCTIONS.md

Everything a new contributor needs to understand, set up, and continue this project.

---

## What This Project Is

A contextual bandit recommendation engine built for GiraXpress — Rwanda's first multi-vendor marketplace. The system treats delivery outcomes as reward signals, not side effects. A LinUCB agent trained with delivery feedback produces better long-term recommendations than one trained on clicks alone.

This repo (`ml-recommendation-system`) is the **standalone research layer**. The production integration lives in [`GiraXpress/ml-service/`](https://github.com/niyibizimadeit/GiraXpress). When files are ready, they get copied from here into GiraXpress — not the other way around.

The thesis argument in one sentence: in markets where delivery reliability varies across sellers, a click-only reward signal trains the model to recommend unreliable sellers. Adding delivery outcome as a signal fixes this, and the fix compounds with catalog curation quality.

---

## Repository Structure

```
ml-recommendation-system/
├── conftest.py                            # Adds src/ to Python path for pytest
├── configs/
│   └── config.yaml                        # All experiment parameters live here
├── docs/
│   ├── adr/                               # Architecture Decision Records
│   │   ├── 001-linucb-design.md
│   │   ├── 002-reward-signal-design.md
│   │   └── 003-ab-split-design.md
│   └── *.png                              # Charts saved by notebooks
├── notebooks/
│   ├── 01_baseline_linucb.ipynb           # LinUCB vs Greedy — gate check
│   ├── 02_alpha_sensitivity.ipynb         # Calibrate alpha — gate check
│   ├── 03_delivery_signal_ablation.ipynb  # Claim 1 — delivery reward effect
│   └── 04_catalog_curation_effect.ipynb   # Claim 2 — curation effect
├── src/
│   ├── bandits/
│   │   ├── __init__.py
│   │   ├── base.py                        # Abstract base class all bandits implement
│   │   ├── linucb.py                      # LinUCB — thesis treatment arm
│   │   └── greedy.py                      # Greedy baseline — A/B control arm
│   ├── data/
│   │   ├── __init__.py
│   │   └── synthetic_generator.py         # Kigali-calibrated interaction stream generator
│   ├── evaluation/                        # NOT YET BUILT — see TODO.md
│   │   ├── regret.py
│   │   ├── ndcg.py
│   │   └── ab_analysis.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── context_builder.py             # Assembles 18-dim context vectors
│   │   └── normalizer.py                  # MinMax and ZScore normalizers
│   └── api/                               # NOT YET BUILT — see TODO.md
│       ├── main.py
│       ├── rank.py
│       └── reward.py
└── tests/
    ├── test_bandits.py                    # 45 tests — all passing
    ├── test_evaluation.py                 # NOT YET BUILT
    └── test_features.py                   # 44 tests — all passing
```

---

## Setup

**Prerequisites:** Python 3.10+, pip, git.

```bash
git clone https://github.com/niyibizimadeit/ml-recommendation-system.git
cd ml-recommendation-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Verify the setup:**

```bash
pytest tests/ -v
```

89 tests should pass (45 bandit + 44 feature). Zero failures means the environment is correct.

---

## How to Run Each File

### Synthetic generator — smoke test
```bash
python src/data/synthetic_generator.py
```
Prints event counts, cohort split, and a validation report. All checks should pass.

### Synthetic generator — from a notebook or script
```python
import yaml
from src.data.synthetic_generator import KigaliSyntheticGenerator

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

gen = KigaliSyntheticGenerator(config, seed=42)
interactions, users, products = gen.generate()
report = gen.validate(interactions, products)
```

### LinUCB — quick check
```python
import numpy as np
from src.bandits.linucb import LinUCB
from src.features.context_builder import build_context, N_FEATURES
from datetime import datetime

model = LinUCB(n_features=N_FEATURES, alpha=1.0)
ctx = build_context(
    timestamp=datetime.now(), device_type="mobile",
    category_affinity={"electronics": 0.7, "accessories": 0.3},
    session_depth=2, price_tier=0.4, product_category="electronics",
    seller_quality_score=0.85, days_since_listed=0.1,
    seller_delivery_reliability=0.9,
)
model.log("product_abc", ctx, reward=20.0)
model.flush()
print(model.rank(["product_abc", "product_xyz"], ctx))
```

### Run all tests
```bash
pytest tests/ -v
```

### Run notebooks
```bash
jupyter notebook notebooks/
```
Open each notebook and run all cells top to bottom with Shift+Enter. Run them in order — notebook 02 depends on the simulation setup from notebook 01.

### Save and load a model snapshot
```python
model.flush()                           # always flush before saving
model.save("data/snapshots/linucb.pkl")

from src.bandits.linucb import LinUCB
model = LinUCB.load("data/snapshots/linucb.pkl")
```

---

## Architecture — How the Pieces Connect

### The context vector (18 features)

Every recommendation call builds one context vector. `context_builder.py` assembles it. LinUCB must always be initialized with `n_features=18`.

```
Index  Feature                         Source
─────────────────────────────────────────────────────────
0      time_of_day                     request timestamp
1      device_mobile                   user agent / cookie
2      device_desktop                  user agent / cookie
3-7    affinity_{category}             interaction history (5 categories)
8      session_depth                   current session page count
9      price_tier                      product record
10-14  category_{name}                 product record one-hot (5 categories)
15     seller_quality_score            seller record
16     days_since_listed               product record
17     seller_delivery_reliability     seller record — Phase 15 signal
```

### Session-level updates

LinUCB buffers interactions via `log()` and applies them all at session end via `flush()`. Never update per-event in production — it creates threading problems and is slower.

```python
# During a session — buffer interactions
for product_id, ctx, reward in session_events:
    model.log(product_id, ctx, reward)

# At session end — apply all at once
model.flush()
```

### Cold start logic

| User history | Policy used |
|---|---|
| 0 interactions | Stage 1: trending / bestsellers |
| 1–9 interactions | Stage 2: scoring model (category affinity + popularity) |
| 10+ interactions | Stage 3: LinUCB |

10 total interaction rows in the `interactions` table for that `user_id` is the trigger. Guests always use Stage 1.

### A/B cohort assignment

- Logged-in users: hash `user_id` mod 5. Result 0 → Greedy (20%). Result 1–4 → LinUCB (80%).
- Guests: random assignment at session start, stored in cookie.
- Cohort is sticky — a user assigned to LinUCB stays in LinUCB across sessions.
- Every interaction row records `served_by: 'linucb' | 'greedy'`.

### Model persistence

In-memory during requests. Periodic cron snapshot to disk using `model.save()`. Never save with a non-empty buffer — `flush()` first. The save will raise `RuntimeError` if you forget.

---

## Key Design Decisions

**Why disjoint LinUCB (one A matrix per arm)?**
Hybrid LinUCB shares a global feature matrix across arms — better for very sparse catalogs. Disjoint is simpler, easier to debug, and correct for a catalog of 50–200 products. See `docs/adr/001-linucb-design.md`.

**Why delivery reward is asymmetric (−10 failure, +3 success)?**
A failed delivery destroys trust, generates a refund, and permanently damages the user's perception of the platform. A successful delivery is the expected baseline. The −10/+3 ratio reflects this asymmetry. See `docs/adr/002-reward-signal-design.md`.

**Why no Docker for local development?**
Supabase is hosted. The ML service runs locally with `uvicorn`. No Docker dependency keeps onboarding under 5 minutes.

**Why session-level updates instead of per-event?**
Simpler threading model in a FastAPI server. Per-event updates require locking the A matrix on every request. Session-level batches the writes and releases the lock once per session.

---

## Config File

All experiment parameters live in `configs/config.yaml`. Change values here — not in source code.

```yaml
bandit:
  alpha: 1.0                    # Set this after running notebook 02
  update_frequency: "session"
  arms: 50

rewards:
  click: 1
  add_to_cart: 5
  purchase: 20
  delivery_success: 3
  delivery_failure: -10
  delivery_lambda: 1.0          # λ weight for delivery-adjusted reward

ab_split:
  treatment_ratio: 0.80         # LinUCB
  control_ratio: 0.20           # Greedy

simulation:
  n_users: 500
  n_products: 200
  n_rounds: 10000
  curation_level: 1.0           # 1.0 = fully curated
```

**Critical:** `rewards` values in `config.yaml` must stay in sync with `GiraXpress/src/constants/rewards.ts`. If you change a reward value here, change it there too.

---

## Relation to GiraXpress

When files in this repo are ready, they are copied into GiraXpress — not imported or symlinked.

| File in this repo | Destination in GiraXpress |
|---|---|
| `src/bandits/linucb.py` | `GiraXpress/ml-service/app/bandits/linucb.py` |
| `src/bandits/greedy.py` | `GiraXpress/ml-service/app/bandits/greedy.py` |
| `src/api/rank.py` | `GiraXpress/ml-service/app/api/rank.py` |
| `src/api/reward.py` | `GiraXpress/ml-service/app/api/reward.py` |
| `src/features/context_builder.py` | `GiraXpress/ml-service/app/features/context_builder.py` |

The GiraXpress ML service uses the same context vector schema and reward constants. Keep `configs/config.yaml` and `GiraXpress/src/constants/rewards.ts` in sync manually.

---

## Conventions

- Never edit `src/types/database.types.ts` in GiraXpress by hand — it is auto-generated.
- Never put server data in Zustand (GiraXpress side). TanStack Query owns all server state.
- Never expose the ML service URL to the browser. All ML calls go through `/api/recommendations` in Next.js.
- Delivery outcome is a reward signal. Write it to the `interactions` table — never swallow it.
- `flush()` before `save()`. Always.
- Change config values in `config.yaml`, not in source files.