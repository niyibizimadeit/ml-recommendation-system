# ml-recommendation-system

A contextual bandit recommendation engine built for sparse-data, mobile-first e-commerce environments. Developed as the ML research layer of [GiraXpress](https://github.com/niyibizimadeit/GiraXpress) — Rwanda's first feedback-aware marketplace — and written as a standalone research contribution.

The core argument: delivery outcomes are reward signals, not side effects. A LinUCB agent trained with delivery feedback produces better long-term recommendations than one trained on clicks alone. This repo implements, evaluates, and ablates that claim.

---

## Research Contribution

Three claims, each testable:

1. **Delivery-adjusted reward improves recommendation quality.** Including delivery outcome (success/failure) as a first-class reward signal in LinUCB training reduces long-term regret compared to click-signal-only training.

2. **Catalog curation accelerates bandit convergence.** A curated catalog (fewer noisy listings) produces cleaner interaction data, which reduces the number of rounds LinUCB needs to converge. This is measurable via regret curves across simulated curation levels.

3. **Contextual bandits outperform greedy baselines in sparse-data markets.** In low-volume environments like Kigali's early-stage e-commerce, LinUCB's exploration bonus matters more than in high-volume markets where greedy policies converge faster.

---

## Folder Structure

```
ml-recommendation-system/
├── configs/
│   └── config.yaml                    # Alpha, reward weights, A/B split ratio, update frequency
│
├── data/
│   ├── raw/                           # Raw interaction exports (gitignored)
│   ├── processed/                     # Normalized feature matrices
│   └── logs/
│       └── interactions.db            # SQLite log for local simulation runs
│
├── docs/
│   ├── adr/
│   │   ├── 001-linucb-design.md       # Why LinUCB over Thompson Sampling or UCB1
│   │   ├── 002-reward-signal-design.md # Reward scale rationale and delivery weight λ
│   │   └── 003-ab-split-design.md     # A/B cohort design and measurement plan
│   └── paper/                         # Draft paper assets
│
├── notebooks/
│   ├── 01_baseline_linucb.ipynb       # LinUCB vs Greedy on synthetic data
│   ├── 02_alpha_sensitivity.ipynb     # Regret curves across α ∈ {0.1, 0.5, 1.0, 2.0}
│   ├── 03_delivery_signal_ablation.ipynb  # With vs without delivery reward
│   └── 04_catalog_curation_effect.ipynb  # Regret under curation levels 20%–100%
│
├── src/
│   ├── api/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── rank.py                    # POST /rank — returns ranked product list
│   │   └── reward.py                  # POST /reward — logs interaction, updates bandit
│   │
│   ├── bandits/
│   │   ├── base.py                    # Abstract base class all bandits implement
│   │   ├── linucb.py                  # LinUCB (thesis treatment arm)
│   │   ├── greedy.py                  # Greedy baseline (A/B control arm)
│   │   └── thompson_sampling.py       # Thompson Sampling (comparison)
│   │
│   ├── data/
│   │   ├── synthetic_generator.py     # Generates realistic interaction streams for simulation
│   │   └── kigali_product_profiles.py # Product feature distributions from Kigali market
│   │
│   ├── evaluation/
│   │   ├── regret.py                  # Cumulative and per-round regret computation
│   │   ├── ndcg.py                    # NDCG@K per cohort
│   │   └── ab_analysis.py             # Cohort comparison: conversion, reward, regret
│   │
│   ├── features/
│   │   ├── context_builder.py         # Assembles user + product context vectors
│   │   └── normalizer.py              # Feature normalization (min-max, z-score)
│   │
│   ├── ingestion/
│   │   └── event_tracker.py           # Receives and logs interaction events locally
│   │
│   └── models/
│       └── schemas.py                 # Pydantic models for all API payloads
│
└── tests/
    ├── test_bandits.py                # Unit tests: LinUCB update, arm selection
    ├── test_evaluation.py             # Unit tests: regret, NDCG correctness
    └── test_features.py               # Unit tests: context vector shape, normalization
```

---

## The Recommendation System

### Three-stage architecture

**Stage 1 — Heuristic (cold start)**
No interaction history available. Returns trending products and bestsellers. Used for new users and as fallback when the bandit model is unavailable.

**Stage 2 — Scoring model**
Deterministic ranking combining category affinity, product popularity, seller quality score, and listing recency. Used when a user has limited history (fewer than 10 interactions).

**Stage 3 — LinUCB contextual bandit**
The main treatment. Treats each product recommendation as an arm and learns which arms maximize reward per user context. Runs online — the model updates after every logged interaction.

---

### LinUCB

LinUCB (Linear Upper Confidence Bound) models expected reward as a linear function of a context vector. For each arm `a` at round `t`:

$$a_t = \arg\max_{a \in \mathcal{A}} \left( \theta_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t} \right)$$

- `x_t` — context vector (user + product features at round t)
- `θ_a` — learned weight vector for arm a
- `A_a` — regularized feature covariance matrix for arm a
- `α` — exploration coefficient (calibrated via sensitivity analysis, see `notebooks/02_alpha_sensitivity.ipynb`)

The second term is the confidence bonus. High uncertainty about an arm's reward produces a larger bonus, which drives exploration. As more interactions accumulate, the matrix `A_a` fills in and the bonus shrinks — exploitation takes over.

---

### Context vector

Every recommendation call assembles a context vector from two sources:

**User features**
| Feature | Type | Notes |
|---------|------|-------|
| `time_of_day` | Float [0, 1] | Normalized hour |
| `device_type` | One-hot | Mobile / Desktop |
| `category_affinity` | Float [0, 1] per category | Derived from interaction history |
| `session_depth` | Int | Pages viewed in current session |

**Product features**
| Feature | Type | Notes |
|---------|------|-------|
| `price_tier` | Float [0, 1] | Normalized within category |
| `category` | One-hot | Product category |
| `seller_quality_score` | Float [0, 1] | Based on delivery reliability + review score |
| `days_since_listed` | Float [0, 1] | Normalized, capped at 90 days |
| `seller_delivery_reliability` | Float [0, 1] | Added in Phase 15 (delivery feedback loop) |

All continuous features are normalized before being passed to the model. Raw values fed into the dot product let high-magnitude features (price, days) dominate — normalization prevents this.

---

### Reward signals

| Event | Reward | Rationale |
|-------|--------|-----------|
| No action | 0 | Baseline |
| Click | +1 | Weak positive signal |
| Add to cart | +5 | Strong purchase intent |
| Purchase | +20 | Confirmed conversion |
| Delivery successful | +3 | Delivery-adjusted signal (Phase 15) |
| Delivery failed | −10 | Corrupted user experience + lost trust |

The delivery signals are the research contribution. A product with high click-through but persistent delivery failures receives net negative reward over time — the model learns to deprioritize it. A click-signal-only model cannot learn this.

**Delivery-adjusted reward:**

$$r_t^{\text{adj}} = r_t^{\text{click}} + \lambda \cdot r_t^{\text{delivery}}$$

`λ` is a weighting hyperparameter. The ablation study in `notebooks/03_delivery_signal_ablation.ipynb` sweeps `λ ∈ {0, 0.5, 1.0, 2.0}` to find the optimal balance.

---

### A/B split design

Every session is assigned to one cohort at start and the assignment is stored in a cookie.

| Cohort | Policy | Traffic |
|--------|--------|---------|
| Treatment | LinUCB | 80% |
| Control | Greedy (bestsellers) | 20% |

Every interaction row records `served_by: 'linucb' | 'greedy'`. Metrics computed per cohort:

- **Cumulative regret** — total missed reward relative to oracle policy
- **NDCG@K** — ranking quality at positions 1, 5, 10
- **Click-through rate** — clicks / impressions per session
- **Conversion rate** — purchases / sessions
- **Delivery-adjusted reward** — total reward including delivery outcomes

---

## Relation to GiraXpress

This repo is the standalone research layer. The production integration lives in [`GiraXpress/ml-service/`](https://github.com/niyibizimadeit/GiraXpress).

| Concern | This repo | GiraXpress |
|---------|-----------|------------|
| Algorithm development | ✓ | |
| Simulation and evaluation | ✓ | |
| Paper writing | ✓ | |
| Production API endpoints | ✓ (mirrored) | ✓ (source of truth) |
| Live A/B experiment | | ✓ |
| Supabase integration | | ✓ |

`src/bandits/linucb.py` and `src/api/` are designed to be copied directly into `GiraXpress/ml-service/app/` without modification. The context vector schema and reward constants in `configs/config.yaml` must stay in sync with `GiraXpress/src/constants/rewards.ts`.

---

## Setup

```bash
git clone https://github.com/niyibizimadeit/ml-recommendation-system.git
cd ml-recommendation-system

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Run the FastAPI server locally:**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Run the simulation notebooks:**

```bash
jupyter notebook notebooks/
```

**Run tests:**

```bash
pytest tests/
```

---

## API Endpoints

### `POST /rank`

Returns a ranked list of product IDs for a given user context.

**Request:**
```json
{
  "user_id": "uuid | null",
  "session_id": "string",
  "context": {
    "time_of_day": 0.58,
    "device_type": "mobile",
    "category_affinity": { "electronics": 0.8, "clothing": 0.2 },
    "session_depth": 3
  },
  "candidate_products": [
    {
      "product_id": "uuid",
      "price_tier": 0.4,
      "category": "electronics",
      "seller_quality_score": 0.85,
      "days_since_listed": 0.1
    }
  ]
}
```

**Response:**
```json
{
  "ranked_product_ids": ["uuid-1", "uuid-3", "uuid-2"],
  "served_by": "linucb",
  "cohort": "treatment"
}
```

---

### `POST /reward`

Logs an interaction event and updates the bandit model.

**Request:**
```json
{
  "session_id": "string",
  "product_id": "uuid",
  "event": "purchase",
  "reward": 20,
  "served_by": "linucb",
  "context": { ... }
}
```

**Response:**
```json
{
  "status": "ok",
  "model_updated": true
}
```

---

## Evaluation

### Regret

Cumulative regret measures total reward missed relative to an oracle that always picks the best arm:

$$R_T = \sum_{t=1}^{T} r_{t}^* - r_t$$

Lower is better. LinUCB regret should sublinear — it grows slower than T as the model converges.

### NDCG@K

Normalized Discounted Cumulative Gain measures ranking quality. A purchase in position 1 contributes more than one in position 5.

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Computed separately per cohort. LinUCB should outperform Greedy on NDCG@10 after sufficient training rounds.

### Delivery-adjusted reward

A new metric introduced in this research. Compares total cohort reward computed with and without the delivery signal component. The gap shows how much recommendation quality degrades when the delivery feedback loop is removed.

---

## Build Sequence

Start here:

1. `src/data/synthetic_generator.py` — build the simulation engine first. Without realistic synthetic data you cannot tune the model before real users arrive.
2. `src/bandits/linucb.py` — implement and unit-test the core algorithm.
3. `src/features/context_builder.py` and `src/features/normalizer.py` — wire the feature pipeline.
4. `notebooks/01_baseline_linucb.ipynb` — validate LinUCB vs Greedy on synthetic data.
5. `notebooks/02_alpha_sensitivity.ipynb` — fix alpha before building the API.
6. `src/api/rank.py` and `src/api/reward.py` — expose the model over HTTP.
7. Copy `src/bandits/linucb.py` and `src/api/` into GiraXpress `ml-service/`.

---

## Paper Scope

The standalone paper covers the recommendation system in depth, independent of the broader GiraXpress thesis.

**Working title:** *Delivery-aware contextual bandits for sparse-data e-commerce recommendation*

**Core claims:**
- Delivery outcome as a reward signal reduces long-term recommendation regret
- Catalog curation quality directly affects bandit convergence speed
- LinUCB outperforms greedy policies in low-volume emerging market conditions

**Evaluation:**
- Simulation study using synthetic Kigali interaction streams
- Ablation: delivery reward weight λ ∈ {0, 0.5, 1.0, 2.0}
- Ablation: catalog curation level 20% → 100% clean listings
- Comparison: LinUCB vs Greedy vs Thompson Sampling on regret and NDCG@K

---

## Configuration

`configs/config.yaml` controls all experiment parameters. Change values here, not in source code.

```yaml
bandit:
  alpha: 1.0                    # Exploration coefficient — tune via notebook 02
  update_frequency: "session"   # "event" | "session" | "batch"
  arms: 50                      # Max products ranked per call

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
  curation_level: 1.0           # 1.0 = fully curated, 0.2 = 20% clean listings
```

---

*Research component of [GiraXpress](https://github.com/niyibizimadeit/GiraXpress). Built in Kigali, Rwanda.*