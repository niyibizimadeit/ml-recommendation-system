# Delivery-Aware Contextual Bandits for E-Commerce Recommendation

> **Research project · Reinforcement Learning · Contextual Bandits · Emerging Markets**

A contextual bandit recommendation system that treats delivery outcomes as first-class reward signals. Developed as the ML research layer of [GiraXpress](https://github.com/niyibizimadeit/GiraXpress) and written as a standalone, reproducible research contribution.

The core argument: in markets where delivery reliability varies across sellers, a model trained on click signals alone learns the wrong thing. A delivery-aware reward function produces better long-term recommendation quality — and the advantage is largest in sparse-data, low-volume environments where wrong signals are hardest to correct.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-in_progress-yellow.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-forthcoming-red.svg)]()

---

## Research Motivation

Standard recommendation systems optimize for short-horizon signals: clicks, add-to-cart events, purchases. In mature markets with reliable fulfilment infrastructure this works acceptably — delivery failures are rare noise. In emerging-market e-commerce, where delivery reliability varies significantly across informal sellers and informal logistics networks, this assumption breaks down. A product can have high click-through rates and persistent delivery failures simultaneously. A click-signal-only model cannot distinguish between these sellers and will keep recommending the unreliable one.

This project studies whether incorporating delivery outcome as a reward signal in a contextual bandit recommendation system reduces long-term regret, and whether the effect size is meaningful in the low-volume, sparse-data conditions characteristic of early-stage emerging-market platforms.

The research is conducted primarily in simulation — using realistic synthetic interaction streams calibrated to Kigali market conditions — with a live A/B validation component in [GiraXpress](https://github.com/niyibizimadeit/GiraXpress) as the deployment environment.

---

## Research Contributions

Three claims, each testable, each with a corresponding ablation notebook:

**Claim 1 — Delivery-adjusted reward reduces long-term recommendation regret.**
LinUCB trained with a delivery-outcome-augmented reward function achieves lower cumulative regret than LinUCB trained on click signals alone, in simulation over 10,000 rounds. The effect is measured via ablation over delivery reward weight λ ∈ {0, 0.5, 1.0, 2.0}.

**Claim 2 — Catalog curation quality directly affects bandit convergence speed.**
A curated catalog (low proportion of noisy, unreliable-seller listings) produces cleaner interaction data, which reduces the number of rounds LinUCB needs to converge. Measured by running the simulation across curation levels from 20% to 100% clean listings and observing regret curves.

**Claim 3 — LinUCB outperforms greedy baselines in sparse-data, low-volume conditions.**
LinUCB's exploration bonus matters more in low-volume markets (< 500 users) than in high-volume ones where greedy policies converge faster. We show this via regret curves across simulated market sizes, establishing the conditions under which contextual bandits provide the largest advantage.

---

## Algorithm

### LinUCB — Linear Upper Confidence Bound

LinUCB models the expected reward for each recommendation arm as a linear function of a context vector. At each round *t*, it selects the arm that maximizes the sum of expected reward and an exploration bonus:

$$a_t = \arg\max_{a \in \mathcal{A}} \left( \theta_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t} \right)$$

where:
- $x_t \in \mathbb{R}^d$ — context vector (user + product features at round *t*)
- $\theta_a = A_a^{-1} b_a$ — learned weight vector for arm *a*
- $A_a = I + \sum x_s x_s^T$ — regularized feature covariance matrix for arm *a*
- $b_a = \sum r_s x_s$ — reward-weighted feature accumulator
- $\alpha$ — exploration coefficient (calibrated via sensitivity analysis in `notebooks/02_alpha_sensitivity.ipynb`)

The second term $\alpha \sqrt{x_t^T A_a^{-1} x_t}$ is the confidence bonus. High uncertainty about an arm's true reward produces a larger bonus, driving exploration. As interactions accumulate, $A_a$ fills in and the bonus shrinks — exploitation takes over. LinUCB has a theoretical regret bound of $O(\sqrt{dT \log T})$, sublinear in the number of rounds *T*. The simulation studies verify this bound holds empirically under Kigali market conditions.

### Why LinUCB over alternatives

Thompson Sampling requires a prior over reward distributions, which is difficult to specify correctly in a sparse-data setting. UCB1 treats all arms as equal and ignores context — it would rank a new high-quality product identically to a new low-quality one. LinUCB uses context (price tier, seller reliability, product category) to inform its uncertainty estimates, making it better suited to cold-start conditions in a new marketplace. The design rationale is documented in `docs/adr/001-linucb-design.md`.

---

## Context Vector

Every recommendation call assembles a context vector from user and product features:

**User features**

| Feature | Type | Notes |
|---------|------|-------|
| `time_of_day` | Float [0, 1] | Normalized hour |
| `device_type` | One-hot | Mobile / Desktop |
| `category_affinity` | Float [0, 1] per category | Derived from interaction history |
| `session_depth` | Float [0, 1] | Pages viewed this session, normalized |

**Product features**

| Feature | Type | Notes |
|---------|------|-------|
| `price_tier` | Float [0, 1] | Normalized within category |
| `category` | One-hot | Product category |
| `seller_quality_score` | Float [0, 1] | Review score + delivery reliability composite |
| `days_since_listed` | Float [0, 1] | Normalized, capped at 90 days |
| `seller_delivery_reliability` | Float [0, 1] | Running delivery success rate per seller |

All continuous features are normalized before the dot product. Feature normalization method (min-max vs. z-score by feature type) is documented in `src/features/normalizer.py`.

---

## Reward Function

### Base reward (click-signal only)

| Event | Reward |
|-------|--------|
| No action | 0 |
| Click | +1 |
| Add to cart | +5 |
| Purchase | +20 |

### Delivery-adjusted reward

$$r_t^{\text{adj}} = r_t^{\text{click}} + \lambda \cdot r_t^{\text{delivery}}$$

| Event | $r_t^{\text{delivery}}$ |
|-------|------------------------|
| Delivery successful | +3 |
| Delivery failed | −10 |

`λ` is the delivery reward weight hyperparameter. `notebooks/03_delivery_signal_ablation.ipynb` sweeps λ ∈ {0, 0.5, 1.0, 2.0}. Setting λ = 0 recovers the baseline click-only model. The asymmetric magnitude (−10 for failure, +3 for success) reflects that a failed delivery destroys user trust and generates a refund or dispute, while a successful delivery is the baseline expectation. Rationale in `docs/adr/002-reward-signal-design.md`.

---

## Simulation Study

Because GiraXpress is an early-stage platform, the primary evidence for all three claims comes from simulation. The simulation is calibrated to realistic Kigali market conditions:

- **Product catalog** — 200 products with category distribution, price tiers, and seller quality scores informed by starlordgroup.rw order data
- **User behavior model** — click probability as a logistic function of context-reward alignment; add-to-cart and purchase probabilities conditioned on click
- **Delivery outcome model** — delivery success probability per seller sampled from a Beta distribution, with mean and variance estimated from real delivery data
- **Curation model** — fraction of reliable-seller listings varied from 20% to 100% for Claim 2

All random seeds are fixed. Simulation parameters are in `configs/config.yaml`.

---

## A/B Experiment Design

The live A/B experiment in GiraXpress provides held-out validation after sufficient traffic accumulates.

| Cohort | Policy | Traffic share |
|--------|--------|---------------|
| Treatment | LinUCB (delivery-aware) | 80% |
| Control | Greedy (bestsellers) | 20% |

Cohort assignment is made at session start and persisted in a cookie. Every interaction row records `served_by: 'linucb' | 'greedy'`. Metrics per cohort:

- Cumulative regret
- NDCG@K (K ∈ {1, 5, 10})
- Click-through rate
- Conversion rate
- Delivery-adjusted reward (with and without λ component)

A/B design and power analysis are documented in `docs/adr/003-ab-split-design.md`.

---

## Evaluation

### Regret

$$R_T = \sum_{t=1}^{T} r_t^* - r_t$$

Lower is better. LinUCB's $O(\sqrt{dT \log T})$ guarantee means regret grows sublinearly — the per-round average loss decreases over time as the model learns.

### NDCG@K

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

Computed separately per cohort. LinUCB is expected to outperform Greedy on NDCG@10 after sufficient training rounds.

### Simulation Results (Claim 1 — delivery reward ablation)

| λ value | Total regret at T=10,000 | Convergence round |
|---------|--------------------------|-------------------|
| 0.0 (click only) | — | — |
| 0.5 | — | — |
| 1.0 | — | — |
| 2.0 | — | — |

### Simulation Results (Claim 2 — curation effect)

| Curation level | Total regret at T=10,000 |
|----------------|--------------------------|
| 20% clean | — |
| 50% clean | — |
| 80% clean | — |
| 100% clean | — |

*Results populated after simulation notebooks complete.*

---

## Repository Structure

```
ml-recommendation-system/
├── configs/
│   └── config.yaml
│
├── docs/adr/
│   ├── 001-linucb-design.md
│   ├── 002-reward-signal-design.md
│   └── 003-ab-split-design.md
│
├── notebooks/
│   ├── 01_baseline_linucb.ipynb           # LinUCB vs Greedy
│   ├── 02_alpha_sensitivity.ipynb         # α calibration
│   ├── 03_delivery_signal_ablation.ipynb  # Claim 1
│   └── 04_catalog_curation_effect.ipynb   # Claim 2
│
├── src/
│   ├── bandits/
│   │   ├── base.py
│   │   ├── linucb.py                      # Treatment arm
│   │   ├── greedy.py                      # Control arm
│   │   └── thompson_sampling.py           # Comparison
│   ├── data/
│   │   ├── synthetic_generator.py
│   │   └── kigali_product_profiles.py
│   ├── evaluation/
│   │   ├── regret.py
│   │   ├── ndcg.py
│   │   └── ab_analysis.py
│   ├── features/
│   │   ├── context_builder.py
│   │   └── normalizer.py
│   └── api/
│       ├── main.py
│       ├── rank.py                        # POST /rank
│       └── reward.py                      # POST /reward
│
└── tests/
    ├── test_bandits.py
    ├── test_evaluation.py
    └── test_features.py
```

---

## Build Sequence

1. `src/data/synthetic_generator.py` — build the simulation engine first
2. `src/bandits/linucb.py` — implement and unit-test the algorithm
3. `src/features/context_builder.py` + `normalizer.py` — wire feature pipeline
4. `notebooks/01_baseline_linucb.ipynb` — validate LinUCB vs Greedy
5. `notebooks/02_alpha_sensitivity.ipynb` — calibrate alpha
6. `notebooks/03_delivery_signal_ablation.ipynb` — Claim 1
7. `notebooks/04_catalog_curation_effect.ipynb` — Claim 2
8. `src/api/` — expose model over HTTP
9. Mirror into GiraXpress `ml-service/`

---

## Setup

```bash
git clone https://github.com/niyibizimadeit/ml-recommendation-system.git
cd ml-recommendation-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

uvicorn src.api.main:app --reload --port 8000
jupyter notebook notebooks/
pytest tests/
```

---

## Companion Paper

**Working title:** *Delivery-Aware Contextual Bandits for Sparse-Data E-Commerce Recommendation in Emerging Markets*

**Target venues:** RecSys 2027 · NeurIPS 2027 RL workshop · ICML 2027 workshop

**Abstract (draft):** We study whether incorporating delivery outcome as a reward signal in a LinUCB contextual bandit recommendation system reduces long-term regret in emerging-market e-commerce settings characterized by sparse interaction data and variable seller delivery reliability. We introduce a delivery-adjusted reward function and evaluate its effect via simulation over realistic Kigali market interaction streams, with ablation over delivery reward weight λ ∈ {0, 0.5, 1.0, 2.0} and catalog curation quality from 20% to 100% clean listings. We show that delivery-aware training reduces cumulative regret, that the effect is amplified in sparse-data conditions, and that catalog curation directly affects bandit convergence speed. We validate these findings using a live A/B experiment on GiraXpress, Rwanda's first feedback-aware marketplace. All simulation code and experiment configurations are publicly released.

---

## Project Status

| Component | Status |
|-----------|--------|
| LinUCB, Greedy, Thompson Sampling | 🔄 In progress |
| Simulation engine | 🔄 In progress |
| Feature pipeline | ⏳ Planned |
| Notebook 01 — baseline | ⏳ Planned |
| Notebook 02 — alpha sensitivity | ⏳ Planned |
| Notebook 03 — delivery signal ablation | ⏳ Planned |
| Notebook 04 — curation effect | ⏳ Planned |
| FastAPI endpoints | ⏳ Planned |
| GiraXpress integration (Phase 12) | ⏳ Planned |
| Live A/B experiment | ⏳ Planned |
| arXiv preprint | ⏳ Planned |

---

## License

MIT License © 2026 NIYIBIZI Prince

---

*Research layer of [GiraXpress](https://github.com/niyibizimadeit/GiraXpress). Built in Kigali, Rwanda.*
