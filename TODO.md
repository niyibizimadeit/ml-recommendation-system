# TODO.md

Current state, what is done, what is next, and exactly how to continue.

Last updated: Phase ML-4 complete.

---

## Current State

### Done — do not rebuild

| Phase | What was built | Test coverage |
|---|---|---|
| ML-1 | `src/data/synthetic_generator.py` — Kigali-calibrated interaction stream generator | `gen.validate()` — all checks pass |
| ML-2 | `src/bandits/linucb.py`, `greedy.py`, `base.py` | 45 tests passing in `tests/test_bandits.py` |
| ML-3 | `src/features/context_builder.py`, `normalizer.py` | 44 tests passing in `tests/test_features.py` |
| ML-4 | `notebooks/01_baseline_linucb.ipynb`, `notebooks/02_alpha_sensitivity.ipynb` | Run notebooks — verify charts render and gate checks pass |

**Total tests:** 89 passing. Run `pytest tests/ -v` to confirm before continuing.

---

## Immediate Next Step — Before Writing Any Code

Run notebooks 01 and 02 if you have not already:

```bash
jupyter notebook notebooks/
```

Notebook 01 gate: LinUCB cumulative regret curve bends away from Greedy. Per-round regret drops toward 0 while Greedy stays flat.

Notebook 02 gate: identify the alpha with the lowest bar in the final bar chart. Write it into `configs/config.yaml` under `bandit.alpha`. This value must be locked before building the API.

---

## Phase ML-5 — FastAPI Endpoints (Completed)

**Goal:** Expose the LinUCB model over HTTP with two endpoints: `/rank` and `/reward`.

**Files to create:**
- `src/api/main.py` — FastAPI app entry point
- `src/api/rank.py` — `POST /rank` handler
- `src/api/reward.py` — `POST /reward` handler
- `src/models/schemas.py` — Pydantic request/response models

**`POST /rank` — request body:**
```json
{
  "user_id": "uuid | null",
  "session_id": "string",
  "context": {
    "time_of_day": 0.58,
    "device_type": "mobile",
    "category_affinity": { "electronics": 0.8, "accessories": 0.2 },
    "session_depth": 3
  },
  "candidate_products": [
    {
      "product_id": "uuid",
      "price_tier": 0.4,
      "category": "electronics",
      "seller_quality_score": 0.85,
      "days_since_listed": 0.1,
      "seller_delivery_reliability": 0.9
    }
  ]
}
```

**`POST /rank` — response:**
```json
{
  "ranked_product_ids": ["uuid-1", "uuid-3", "uuid-2"],
  "served_by": "linucb",
  "cohort": "treatment"
}
```

**`POST /reward` — request body:**
```json
{
  "session_id": "string",
  "product_id": "uuid",
  "event": "purchase",
  "reward": 20,
  "served_by": "linucb",
  "context": { "..." : "..." }
}
```

**`POST /reward` — response:**
```json
{
  "status": "ok",
  "model_updated": true
}
```

**How `/rank` should work internally:**
1. Determine cohort from `user_id` hash (mod 5 → 0 = Greedy, 1–4 = LinUCB) or random for guests.
2. Build context vector using `context_builder.build()` — one call per candidate product.
3. Call `linucb.rank(candidate_ids, ctx)` or `greedy.rank(candidate_ids)`.
4. Return ranked product IDs and `served_by` value.

**How `/reward` should work internally:**
1. Reconstruct the context vector from the request.
2. Call `linucb.log(product_id, ctx, reward)` — do not flush here.
3. Flush is triggered at session end — implement a `POST /flush` endpoint or flush on a timer.
4. Return `model_updated: true`.

**Model in memory:**
The LinUCB and Greedy instances must be module-level singletons in `main.py`. FastAPI is async — use a lock around `flush()` to prevent concurrent writes corrupting `A` matrices.

```python
import asyncio
linucb = LinUCB(n_features=N_FEATURES, alpha=config["bandit"]["alpha"])
greedy = GreedyBaseline()
model_lock = asyncio.Lock()
```

**Run the API locally:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Gate:** `POST /rank` returns ranked product IDs. `POST /reward` returns `model_updated: true`. Test with `curl` or the FastAPI docs at `http://localhost:8000/docs`.

---

## Phase ML-6 — Ablation Notebooks

**Goal:** Produce the evidence for research Claims 1 and 2.

### Notebook 03 — Delivery signal ablation (Claim 1)

**File:** `notebooks/03_delivery_signal_ablation.ipynb`

**What it measures:** Does adding delivery outcome to the reward signal reduce long-term cumulative regret?

**Method:**
- Run LinUCB 4 times — once per λ value: {0, 0.5, 1.0, 2.0}
- λ = 0 is the baseline (click-only). λ > 0 adds delivery signal.
- Reward formula: `r_adj = r_click + λ * r_delivery`
- Use `curation_level = 1.0` to isolate the delivery signal effect
- 5 seeds per λ value. Report mean and std of cumulative regret at T=10,000.

**Reward values to use (from config.yaml):**
- delivery_success: +3
- delivery_failure: −10

**Expected result:** λ = 1.0 or λ = 0.5 should produce the lowest regret. λ = 0 should be worst because the model cannot distinguish reliable from unreliable sellers.

**Populate this table in the README:**
```
| λ value | Total regret at T=10,000 | Convergence round |
|---------|--------------------------|-------------------|
| 0.0     |                          |                   |
| 0.5     |                          |                   |
| 1.0     |                          |                   |
| 2.0     |                          |                   |
```

**Gate:** λ > 0 produces measurably lower regret than λ = 0.

### Notebook 04 — Catalog curation effect (Claim 2)

**File:** `notebooks/04_catalog_curation_effect.ipynb`

**What it measures:** Does catalog curation level affect how fast LinUCB converges?

**Method:**
- Run LinUCB at curation levels: {0.2, 0.5, 0.8, 1.0}
- `curation_level` is passed to `KigaliSyntheticGenerator(config, seed=42)`
- Use best λ from notebook 03. Use best alpha from notebook 02.
- 5 seeds per curation level. Report mean cumulative regret at T=10,000.
- Also report the round at which per-round regret first drops below 1.0.

**Expected result:** Higher curation level → faster convergence → lower regret. At 20% curation, LinUCB should take significantly more rounds to converge because noise listings dilute the reward signal.

**Populate this table in the README:**
```
| Curation level | Total regret at T=10,000 |
|----------------|--------------------------|
| 20% clean      |                          |
| 50% clean      |                          |
| 80% clean      |                          |
| 100% clean     |                          |
```

**Gate:** Regret at 20% curation is measurably higher than at 100% curation. The convergence round grows as curation decreases.

---

## Phase ML-7 — GiraXpress Integration

**Goal:** Wire the validated model into GiraXpress `ml-service/` and connect it to the Next.js frontend.

**Steps:**

1. Copy files from this repo into GiraXpress:
   ```
   src/bandits/linucb.py      → GiraXpress/ml-service/app/bandits/linucb.py
   src/bandits/greedy.py      → GiraXpress/ml-service/app/bandits/greedy.py
   src/api/rank.py            → GiraXpress/ml-service/app/api/rank.py
   src/api/reward.py          → GiraXpress/ml-service/app/api/reward.py
   src/features/context_builder.py → GiraXpress/ml-service/app/features/context_builder.py
   ```

2. In GiraXpress `src/app/api/recommendations/route.ts`:
   - This is the Next.js proxy to FastAPI `/rank`.
   - Never call FastAPI directly from the browser.
   - The route reads from the `interactions` table to check if the user has 10+ interactions (LinUCB threshold).
   - Passes context to FastAPI, gets ranked product IDs back, fetches those products from Supabase, returns them.

3. In GiraXpress `src/services/tracking.service.ts`:
   - After writing an interaction event to Supabase, also call FastAPI `POST /reward`.
   - Only call `/reward` for events: `click`, `add_to_cart`, `purchase`.
   - Pass the reward value from `rewards.ts` constants.

4. A/B cohort cookie:
   - Assigned at session start in a Next.js middleware or layout.
   - For logged-in users: `hash(user_id) % 5 === 0` → `greedy`, else → `linucb`.
   - For guests: random, stored in cookie named `gx_cohort`.
   - The cohort value is passed in every `/rank` request.

5. Homepage "For You" section:
   - Authenticated users with 10+ interactions: call `/api/recommendations` server-side.
   - Authenticated users with fewer than 10 interactions: use scoring model (category affinity sort).
   - Guests: show trending products from Supabase.

6. Verify `served_by` is populated in the `interactions` table after integration.

**Gate:** Log in as a test user, browse products, verify `served_by` column is populated in Supabase `interactions` table. Check FastAPI logs for model updates.

---

## Remaining Files to Build (not yet started)

### `src/evaluation/regret.py`
Computes cumulative and per-round regret from an interaction log.
Used by notebooks 03 and 04. Should accept a pandas DataFrame of interactions and an oracle reward value.

### `src/evaluation/ndcg.py`
Computes NDCG@K per cohort from a ranked interaction log.
Formula: `NDCG@K = DCG@K / IDCG@K` where `DCG@K = Σ rel_i / log2(i+1)`.

### `src/evaluation/ab_analysis.py`
Cohort comparison: conversion rate, click-through rate, delivery-adjusted reward, per-cohort NDCG.
Reads from the `interactions` DataFrame, splits by `served_by`, and computes metrics per cohort.

### `tests/test_evaluation.py`
Unit tests for regret, NDCG, and AB analysis functions.
Gate: regret of oracle policy = 0. NDCG of perfect ranking = 1.0.

### `src/bandits/thompson_sampling.py`
Thompson Sampling comparison arm — not used in production, only in evaluation notebooks.
Uses Beta distribution priors over binary reward (click/no-click).

### `src/data/kigali_product_profiles.py`
Product feature distributions calibrated to Kigali market data.
Currently the synthetic generator uses uniform distributions. This file should replace them with distributions informed by real starlordgroup.rw order data when available.

### `docs/adr/001-linucb-design.md`
Why LinUCB over Thompson Sampling and UCB1. Documents the design rationale.

### `docs/adr/002-reward-signal-design.md`
Delivery reward asymmetry rationale. Why −10/+3 and not a symmetric scale.

### `docs/adr/003-ab-split-design.md`
A/B split design and power analysis. How many users are needed before cohort comparison is statistically valid.

---

## Things That Must Stay in Sync

These pairs of values are defined in two places. Change one, change both.

| This repo | GiraXpress |
|---|---|
| `configs/config.yaml` → `rewards.*` | `GiraXpress/src/constants/rewards.ts` |
| `configs/config.yaml` → `ab_split.*` | `GiraXpress/src/constants/ab.ts` |
| `src/features/context_builder.py` → `N_FEATURES = 18` | `GiraXpress/ml-service/app/features/context_builder.py` |

---

## Known Issues / Watch Points

**Alpha is not yet locked.** Run notebook 02 and update `configs/config.yaml` before building the API. The API reads alpha from config on startup.

**`flush()` threading.** In the FastAPI server, multiple requests can arrive concurrently. Use `asyncio.Lock()` around `model.flush()` to prevent concurrent writes to the A matrices from corrupting the model state. This is documented in the Phase ML-5 section above.

**Session boundary in production.** The simulation flushes at the end of each simulated session. In production, the session ends when the user closes the browser or after a timeout. Use a 30-minute inactivity timeout to trigger flush — implement this as a background task in FastAPI.

**Notebook 01 and 02 must be re-run if `configs/config.yaml` changes.** The notebooks use the config file at runtime. If reward values or simulation parameters change, re-run from cell 1.

**GiraXpress Phase 11 must be complete before integration.** The `interactions` table must exist and be receiving events before Phase ML-7 can wire the feedback loop.