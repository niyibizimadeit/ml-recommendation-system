# Delivery-Aware Contextual Bandits for Sparse-Data E-Commerce Recommendation in Emerging Markets

**NIYIBIZI Prince**
Department of Computer Science, Taizhou University
princeniyibizi4@gmail.com

---

## Abstract

Standard recommendation systems optimize for short-horizon signals — clicks, add-to-cart events, purchases. In mature markets with reliable fulfilment infrastructure this is acceptable: delivery failures are rare noise. In emerging-market e-commerce, where delivery reliability varies significantly across informal sellers and informal logistics networks, this assumption breaks down. A product can simultaneously have high click-through rates and persistent delivery failures. A click-signal-only model cannot distinguish between reliable and unreliable sellers, and will continue recommending the unreliable one.

We study whether incorporating delivery outcome as a reward signal in a LinUCB contextual bandit recommendation system reduces long-term regret in sparse-data, low-volume conditions characteristic of early-stage emerging-market platforms. We introduce a delivery-adjusted reward function $r_t^{\text{adj}} = r_t^{\text{click}} + \lambda \cdot r_t^{\text{delivery}}$ and evaluate its effect via simulation over Kigali-calibrated synthetic interaction streams. An ablation over delivery reward weight $\lambda \in \{0, 0.5, 1.0, 2.0\}$ shows that delivery-aware training reduces cumulative regret by up to 93.8% compared to click-signal-only training, with the model fully abandoning unreliable sellers at $\lambda = 2.0$. We additionally show that LinUCB outperforms a greedy bestseller baseline by a factor of 33× in cumulative regret on the same sparse-data environment. A secondary claim regarding catalog curation effects on bandit convergence speed was investigated but produced inconclusive simulation results; we discuss this limitation and propose a corrected experimental design. All code, simulation configurations, and experiment notebooks are publicly released.

---

## 1. Introduction

E-commerce recommendation in emerging markets presents a configuration of challenges that mature-market systems are not designed for: low transaction volumes, unreliable delivery infrastructure, heterogeneous seller reliability, and sparse per-user interaction histories. Rwandan e-commerce is representative of this configuration. Mobile penetration is high but platform adoption is early-stage, meaning most users have few or no prior interactions when they first arrive. Sellers range from established businesses with professional logistics to informal traders operating without reliable pickup addresses.

Standard recommendation approaches — collaborative filtering, content-based ranking, and greedy bestseller policies — share a structural weakness in this setting: they optimize for signals that are available immediately (clicks, views) while ignoring signals that arrive later and carry more information (delivery outcome). A seller who lists high-appeal products but fails to fulfil orders reliably will generate high click-through rates while destroying user trust. A click-signal-only model learns to recommend this seller more, not less.

This paper studies a specific intervention: augmenting the LinUCB contextual bandit reward signal with delivery outcome. We frame this as a reward engineering problem within the contextual bandit framework and evaluate whether the delivery signal meaningfully shifts model behavior in a simulation calibrated to Kigali market conditions.

The contributions are:

1. A delivery-adjusted reward formulation for contextual bandit recommendation in e-commerce.
2. An ablation study demonstrating that $\lambda = 2.0$ eliminates unreliable seller recommendations, reducing cumulative regret by 93.8% compared to click-only training.
3. Empirical confirmation that LinUCB outperforms greedy policies in sparse-data conditions by 33× in cumulative regret.
4. A calibrated synthetic interaction generator for Kigali e-commerce simulation, released as an open research artifact.
5. An honest account of an inconclusive curation experiment, with a corrected experimental design for future work.

---

## 2. Background and Related Work

### 2.1 Contextual Bandits for Recommendation

The multi-armed bandit problem provides a natural framework for recommendation: each product is an arm, each user interaction is a round, and the system learns which arms maximize reward. The contextual bandit extension (Langford & Zhang, 2007) allows reward to depend on a context vector assembled from user and product features.

LinUCB (Li et al., 2010) is the standard contextual bandit for recommendation. It models expected reward as a linear function of the context vector and maintains a per-arm uncertainty estimate derived from the feature covariance matrix. The selection rule is:

$$a_t = \arg\max_{a \in \mathcal{A}} \left( \theta_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t} \right)$$

where $x_t \in \mathbb{R}^d$ is the context vector, $\theta_a = A_a^{-1} b_a$ are learned weights, and $\alpha$ controls the exploration-exploitation tradeoff. The second term is a confidence bonus that drives exploration of uncertain arms. LinUCB has a regret bound of $O(\sqrt{dT \log T})$, sublinear in the number of rounds $T$.

We choose LinUCB over Thompson Sampling because the latter requires a prior over reward distributions — difficult to specify in a sparse-data setting where little is known about the reward distribution a priori. We choose it over UCB1 because UCB1 ignores context, treating a new high-quality product identically to a new low-quality one.

### 2.2 Reward Engineering in Bandits

Reward shaping (Ng et al., 1999) is the practice of augmenting the observed reward signal to accelerate learning. In recommendation systems, most deployed systems use a weighted combination of engagement signals (Jeunen et al., 2021; Ie et al., 2019). Delivery outcome as a first-class reward signal has not, to our knowledge, been studied in the contextual bandit recommendation literature.

The closest work is on delayed feedback in bandits (Joulani et al., 2013; Pike-Burke et al., 2018), where the reward for an action is observed after a lag. Delivery outcome is a form of delayed feedback: the click reward is immediate, but the delivery reward arrives hours or days later. Our formulation handles this by treating the delivery outcome as an additive component with a separate weight $\lambda$, allowing ablation over the contribution of the delayed signal.

### 2.3 E-Commerce Recommendation in Emerging Markets

Emerging-market e-commerce recommendation is understudied. The closest work is on recommendation under sparse data (Kula, 2015; He et al., 2017), but these systems assume reliable fulfilment. Our setting differs in that delivery reliability is a first-order variable, not a background assumption.

---

## 3. Problem Formulation

### 3.1 Contextual Bandit Setup

We formulate product recommendation as a disjoint LinUCB problem. At each round $t$:

1. The system observes a user context $x_t^{user}$ (device type, time of day, category affinity, session depth).
2. A set of candidate products $\mathcal{A}_t$ is available, each with product features $x_t^{product}(a)$.
3. The system assembles a context vector $x_t(a) = [x_t^{user} \| x_t^{product}(a)] \in \mathbb{R}^{18}$ for each candidate.
4. LinUCB selects arm $a_t$ and observes reward $r_t$.

The 18-dimensional context vector is described in Table 1.

**Table 1: Context vector features**

| Index | Feature | Type | Source |
|-------|---------|------|--------|
| 0 | time_of_day | Float [0,1] | Request timestamp |
| 1–2 | device_type | One-hot | User agent |
| 3–7 | category_affinity | Float [0,1] × 5 | Interaction history |
| 8 | session_depth | Float [0,1] | Current session |
| 9 | price_tier | Float [0,1] | Product record |
| 10–14 | product_category | One-hot × 5 | Product record |
| 15 | seller_quality_score | Float [0,1] | Seller record |
| 16 | days_since_listed | Float [0,1] | Product record |
| 17 | seller_delivery_reliability | Float [0,1] | Seller record |

### 3.2 Reward Function

The base reward uses engagement signals:

| Event | Reward |
|-------|--------|
| No action | 0 |
| Click | +1 |
| Add to cart | +5 |
| Purchase | +20 |

The delivery-adjusted reward augments this with a post-purchase signal:

$$r_t^{\text{adj}} = r_t^{\text{click}} + \lambda \cdot r_t^{\text{delivery}}$$

where $r_t^{\text{delivery}} \in \{-10, +3\}$ for failed and successful delivery respectively, and $\lambda \geq 0$ is a hyperparameter controlling the contribution of the delivery signal. Setting $\lambda = 0$ recovers the click-only baseline.

The asymmetry ($-10$ for failure, $+3$ for success) reflects the asymmetric consequences: a failed delivery destroys user trust, triggers a refund or dispute, and reduces future purchase probability. A successful delivery is the expected baseline — it earns a modest positive signal rather than a large one.

### 3.3 Regret

Cumulative regret measures total reward missed relative to an oracle that always selects the arm maximizing expected adjusted reward:

$$R_T = \sum_{t=1}^{T} r_t^* - r_t$$

Lower cumulative regret indicates faster convergence to the optimal policy.

---

## 4. Simulation Setup

### 4.1 Kigali-Calibrated Synthetic Generator

We build a synthetic interaction generator calibrated to Kigali market conditions (Table 2). The generator produces realistic interaction streams for simulation without requiring live user data.

**Table 2: Kigali simulation calibration**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Mobile device share | 85% | Kigali mobile-first market |
| Session depth | Poisson(λ ≈ 2.5), cap 10 | Sparse emerging-market browsing |
| Category affinity | Dirichlet(α=0.4) | Sparse, peaked preferences |
| Delivery failure rate | 15–45% (noise arms) | Variable seller reliability |
| Activity pattern | Bimodal: 12–14h, 18–22h | Kigali lunch and evening peaks |

### 4.2 Ablation: Arm Setup for Claim 1

The delivery signal ablation requires a product catalog where click signal and delivery reliability are in conflict. We construct 30 arms in three groups:

- **Type A (8 arms):** Reliable sellers. Click reward 8–12, delivery reliability 0.85–0.95.
- **Type B (7 arms):** Popular but unreliable. Click reward 14–18, delivery reliability 0.05–0.25.
- **Type C (15 arms):** Noise. Click reward 0–4, delivery reliability 0.10–0.40.

At $\lambda = 0$, Type B arms appear optimal (highest clicks). At $\lambda > 0$, the delivery penalty modifies the effective reward. The delivery-adjusted expected value of a Type B arm is approximately $16.4 + \lambda \cdot (0.14 \times 3 + 0.86 \times (-10)) \approx 16.4 - 7.76\lambda$, which falls below Type A arms when $\lambda$ is large enough.

### 4.3 Experimental Protocol

Each configuration runs 5 independent seeds over 3,000 rounds. We report mean and standard deviation of cumulative regret at $T = 3000$.

### 4.4 Alpha Calibration

Before ablation experiments, we calibrate $\alpha$ using a sensitivity sweep over $\alpha \in \{0.1, 0.5, 1.0, 1.5, 2.0\}$ on a bimodal 30-arm catalog (5 high-quality arms with expected reward 16–22, 25 mediocre arms with expected reward 0–5), 5 seeds per value. Results in Table 3.

---

## 5. Results

### 5.1 Alpha Calibration (Notebook 02)

**Table 3: Alpha sensitivity — final cumulative regret at T=4000**

| Alpha | Mean final regret | Notes |
|-------|-------------------|-------|
| **0.1** | **lowest** | Best — exploits early, converges fast |
| 0.5 | moderate | |
| 1.0 | moderate | |
| 1.5 | high | |
| 2.0 | highest | Over-explores |

Best alpha: **0.1**. This result is consistent with theory: in a low-volume sparse-data environment, the model has limited capacity for long exploration phases. Low alpha allows it to commit earlier to the high-reward arms it has identified, which is correct behavior when the oracle arm advantage is large (best arm ~21 vs median ~3.6). All subsequent experiments use $\alpha = 0.1$.

### 5.2 Claim 1: Delivery Signal Ablation (Notebook 03)

**Table 4: Lambda ablation — cumulative regret and arm type selection at T=3000**

| Lambda ($\lambda$) | Final cumulative regret | % Type-A selections | % Type-B selections |
|-------------------|------------------------|---------------------|---------------------|
| 0.0 (click only) | 16,032 | 80.0% | 20.0% |
| 0.5 | 4,708 | 80.0% | 20.0% |
| 1.0 | 4,524 | 80.0% | 20.0% |
| **2.0** | **987** | **100.0%** | **0.0%** |

These results strongly confirm Claim 1. At $\lambda = 0$, the model gravitates toward Type B arms (high clicks, unreliable delivery) 20% of the time, producing the highest regret. At $\lambda = 2.0$, the model completely abandons Type B arms — oracle arms are Type A (reliable sellers) at this lambda value, and 100% of selections go there. Regret falls from 16,032 to 987, a **93.8% reduction**.

The transition between $\lambda = 0$ and $\lambda = 2.0$ is non-linear. At $\lambda = 0.5$ and $\lambda = 1.0$ the arm selection distribution does not change (still 80/20), but regret falls significantly (from 16,032 to ~4,600). This is because the delivery reward component modifies the magnitude of reward received without yet flipping the oracle arm identity. At $\lambda = 2.0$ the expected delivery-adjusted value of Type B arms falls below Type A, and the model correctly shifts.

This non-linearity has practical implications: a modest delivery weight reduces regret meaningfully even before it changes which arm type is selected. Practitioners can set $\lambda = 0.5$ as a conservative starting point and tune upward as delivery signal data accumulates.

### 5.3 LinUCB vs Greedy Baseline (Notebook 01)

In a 30-arm bimodal environment over 4,000 rounds, LinUCB achieved cumulative regret of 2,031 versus the greedy baseline's 68,885 — a 33× advantage. This confirms Claim 3: in sparse-data environments where the greedy policy can lock onto a mediocre arm early and never recover, LinUCB's exploration bonus provides a decisive advantage.

### 5.4 Claim 2: Catalog Curation Effect (Notebook 04 — Inconclusive)

**Table 5: Curation level simulation results**

| Curation level | Mean final regret | Convergence round |
|----------------|-------------------|-------------------|
| 0.2 (80% noise) | 0 | 200 |
| 0.5 | 2,031 | 200 |
| 0.8 | 508 | 200 |
| 1.0 (fully curated) | 508 | 200 |

These results do not support Claim 2 as stated. The curation=0.2 result of 0 regret is a simulation artifact: with 80% noise listings, the bandit happened to identify the sparse set of high-quality arms rapidly in all seeds, producing anomalously low regret. The uniform convergence round of 200 for all curation levels reflects that all runs hit the rolling-window boundary rather than a true convergence point — the threshold was too permissive.

We do not claim support for Claim 2 from these results. The curation effect is theoretically sound (Lattimore & Szepesvári, 2020, Section 19) but requires a different experimental design to measure. The corrected design would fix the set of high-quality arms across curation levels (rather than varying their identity), use per-arm regret rather than cumulative regret, and measure convergence using a tighter threshold calibrated to arm reward variance.

---

## 6. Discussion

### 6.1 Why $\lambda = 2.0$ Crosses a Threshold

The oracle arm at $\lambda = 0$ is a Type B arm (delivery-adjusted value $\approx 17.9$ at $\lambda = 0$). At $\lambda = 1.0$ the oracle shifts to Type A (delivery-adjusted value $\approx 14.0$ for best Type A vs $\approx 8.6$ for best Type B). The arm selection distribution does not immediately follow the oracle shift because the bandit has already accumulated evidence favoring Type B arms. At $\lambda = 2.0$ the signal is large enough to overcome the accumulated evidence, and the bandit correctly de-weights Type B.

This suggests a practical calibration principle: $\lambda$ should be set at least large enough to overcome the click-signal advantage of the unreliable seller class. In our simulation, that threshold is between 1.0 and 2.0. In production, this threshold depends on the empirical click-reward gap between reliable and unreliable sellers, which is estimable from historical data.

### 6.2 Cold Start and the Threshold Effect

The 10-interaction threshold before LinUCB activates (below which a heuristic fallback serves recommendations) means the bandit never faces a pure cold-start problem. This design choice trades off some early-period learning for stability. A future direction is online learning from the first interaction using a prior informed by category-level statistics.

### 6.3 Limitations

**Simulation fidelity.** The Kigali calibration uses estimated distributions for click probability and delivery reliability. Real user behavior may differ, particularly in session depth patterns and category preference clustering.

**Delayed delivery signal.** In production, the delivery reward arrives hours to days after the click. Our simulation assumes immediate delivery feedback. The impact of this delay on learning dynamics is not studied here; delayed feedback bandits (Joulani et al., 2013) address this but add implementation complexity.

**Single-product ranking.** The current formulation ranks products independently. Joint ranking effects (showing one product affects click probability on others) are not modeled.

**Inconclusive curation result.** Claim 2 remains unsubstantiated. The corrected experimental design is described above.

---

## 7. Conclusion

We show that incorporating delivery outcome as a reward signal in LinUCB contextual bandit recommendation reduces cumulative regret by up to 93.8% compared to click-signal-only training in a Kigali-calibrated simulation. The key finding is that delivery-aware training eliminates recommendations from high-click but unreliable sellers when the delivery weight $\lambda$ exceeds a threshold determined by the click-reward gap between seller classes. We recommend $\lambda = 2.0$ for deployments with significant seller reliability variance, with the option to reduce to $\lambda = 0.5$ in more homogeneous catalogs.

LinUCB outperforms a greedy bestseller baseline by 33× in cumulative regret on the same sparse-data environment, confirming the value of exploration in low-volume emerging-market conditions.

Claim 2 — that catalog curation accelerates bandit convergence — is theoretically motivated but not supported by our simulation results. A corrected experimental design is proposed for future work.

All simulation code, the synthetic interaction generator, and experiment notebooks are released at [repository URL].

---

## References

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural collaborative filtering. *WWW*.

Ie, E., et al. (2019). Reinforcement learning for slate-based recommender systems. *NeurIPS*.

Jeunen, O., et al. (2021). Revisiting offline evaluation for implicit-feedback recommender systems. *RecSys*.

Joulani, P., Gyorgy, A., & Szepesvári, C. (2013). Online learning under delayed feedback. *ICML*.

Kula, M. (2015). Metadata embeddings for user and item cold-start recommendations. *RecSys Workshop*.

Langford, J., & Zhang, T. (2007). The epoch-greedy algorithm for multi-armed bandits with side information. *NeurIPS*.

Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.

Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW*.

Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations. *ICML*.

Pike-Burke, C., Agrawal, S., Szepesvári, C., & Grunewalder, S. (2018). Bandits with delayed, aggregated anonymous feedback. *ICML*.
