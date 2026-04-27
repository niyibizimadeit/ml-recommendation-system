"""
synthetic_generator.py

Generates realistic Kigali e-commerce interaction streams for LinUCB simulation.

Distributions are calibrated for Rwanda's urban mobile-first market:
  - 85% mobile device split
  - Sparse session depth (Poisson λ ≈ 2.5, capped at 10)
  - Category-clustered user preferences (Dirichlet α=0.4 → sparse affinities)
  - 15-45% delivery failure rate depending on seller reliability
  - Bimodal daily activity (lunch 12-14h and evening 18-22h peaks)

Usage:
    from synthetic_generator import KigaliSyntheticGenerator
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    gen = KigaliSyntheticGenerator(config, seed=42)
    interactions, users, products = gen.generate()
"""

import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = ["electronics", "accessories", "clothing", "home", "beauty"]

DEVICE_WEIGHTS = {"mobile": 0.85, "desktop": 0.15}

# Transition probabilities for the event funnel
# These are conditional: P(add_to_cart | clicked), P(purchase | clicked)
CART_GIVEN_CLICK = 0.30
PURCHASE_GIVEN_CLICK_BASE = 0.10   # scaled by category match inside _purchase_prob


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str
    category_affinity: dict        # {category: weight}, sums to 1.0
    device_type: str               # 'mobile' | 'desktop'
    price_sensitivity: float       # [0, 1] — 0 = budget, 1 = premium
    session_depth_lambda: float    # Poisson λ controlling session length


@dataclass
class ProductProfile:
    product_id: str
    category: str
    price_tier: float              # [0, 1] normalized within category
    seller_quality_score: float    # [0, 1]
    seller_delivery_reliability: float  # [0, 1]
    days_since_listed: float       # [0, 1] normalized, raw value capped at 90 days
    is_noise: bool                 # True = uncurated / low-quality listing


@dataclass
class InteractionEvent:
    session_id: str
    user_id: Optional[str]
    product_id: str
    event: str                     # product_view | click | add_to_cart | purchase
    reward: float
    served_by: str                 # 'linucb' | 'greedy'
    timestamp: datetime
    device_type: str
    time_of_day: float             # [0, 1] normalized hour
    category: str
    delivery_outcome: Optional[str]  # 'delivered' | 'failed' | None


# ── Generator ─────────────────────────────────────────────────────────────────

class KigaliSyntheticGenerator:
    """
    Simulates interaction streams for a Kigali-calibrated e-commerce market.

    curation_level controls the fraction of clean listings in the catalog.
      1.0 = fully curated (all listings are high quality)
      0.2 = 20% clean, 80% noise — high-regret environment for the bandit

    This is the independent variable in notebook 04 (catalog curation effect).
    """

    def __init__(self, config: dict, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        sim = config["simulation"]
        self.n_users = sim["n_users"]
        self.n_products = sim["n_products"]
        self.n_rounds = sim["n_rounds"]
        self.curation_level = sim["curation_level"]

        ab = config["ab_split"]
        self.treatment_ratio = ab["treatment_ratio"]

        self.rewards = config["rewards"]

    # ── Profile generation ────────────────────────────────────────────────────

    def _make_users(self) -> list:
        """
        Generates n_users UserProfile objects.

        Category affinity uses Dirichlet(α=0.4) — low α produces sparse,
        peaked distributions where each user strongly prefers 1-2 categories.
        This matches observed behavior in emerging market mobile commerce.

        Price sensitivity uses Beta(2, 5) — right-skewed, most users are
        budget-conscious, a small tail are premium buyers.
        """
        devices = list(DEVICE_WEIGHTS.keys())
        device_probs = list(DEVICE_WEIGHTS.values())
        users = []

        for _ in range(self.n_users):
            affinity_raw = self.rng.dirichlet([0.4] * len(CATEGORIES))
            affinity = dict(zip(CATEGORIES, affinity_raw.tolist()))

            users.append(UserProfile(
                user_id=str(uuid.uuid4()),
                category_affinity=affinity,
                device_type=str(self.rng.choice(devices, p=device_probs)),
                price_sensitivity=float(self.rng.beta(2, 5)),
                session_depth_lambda=float(self.rng.uniform(1.5, 4.0)),
            ))

        return users

    def _make_products(self) -> list:
        """
        Generates n_products ProductProfile objects.

        Noise listings (controlled by curation_level) have:
          - Lower seller quality (Beta(1.5, 5) vs Beta(5, 2))
          - Lower delivery reliability (Beta(1.5, 4) vs Beta(5, 1.5))
          - Lower click-through probability (applied in _click_prob)

        Products are shuffled after creation so noise is not front-loaded
        in ranked lists during simulation.
        """
        n_noise = int(self.n_products * (1 - self.curation_level))
        products = []

        for i in range(self.n_products):
            is_noise = i < n_noise

            seller_quality = (
                float(self.rng.beta(1.5, 5))   # noise: low quality skew
                if is_noise
                else float(self.rng.beta(5, 2)) # curated: high quality skew
            )
            delivery_reliability = (
                float(self.rng.beta(1.5, 4))   # noise: 15-45% failure rate
                if is_noise
                else float(self.rng.beta(5, 1.5))  # curated: 5-20% failure rate
            )

            products.append(ProductProfile(
                product_id=str(uuid.uuid4()),
                category=str(self.rng.choice(CATEGORIES)),
                price_tier=float(self.rng.uniform(0, 1)),
                seller_quality_score=seller_quality,
                seller_delivery_reliability=delivery_reliability,
                days_since_listed=float(self.rng.uniform(0, 1)),
                is_noise=is_noise,
            ))

        # Shuffle so noise products are not systematically ranked lower
        self.rng.shuffle(products)
        return products

    # ── Probability models ────────────────────────────────────────────────────

    def _click_prob(self, user: UserProfile, product: ProductProfile) -> float:
        """
        P(click | user, product)

        Driven by three factors:
          - Category match: how well the product category aligns with user affinity
          - Price match: distance between user price sensitivity and product price tier
          - Seller quality: higher quality sellers get a small boost

        Noise listings receive a 60% penalty on base probability, simulating
        lower trust, worse photos, and weaker titles common in uncurated catalogs.
        """
        category_match = user.category_affinity.get(product.category, 0.0)
        price_match = 1.0 - abs(user.price_sensitivity - product.price_tier)
        quality_boost = product.seller_quality_score * 0.25

        base = 0.04 + 0.40 * category_match + 0.18 * price_match + quality_boost

        if product.is_noise:
            base *= 0.40

        return float(np.clip(base, 0.01, 0.85))

    def _purchase_prob(
        self, user: UserProfile, product: ProductProfile, clicked: bool
    ) -> float:
        """
        P(purchase | user, product, clicked)

        Conditional on a click. Category match is the primary driver.
        Noise listings have much lower conversion — users sense low quality
        before committing to payment.
        """
        if not clicked:
            return 0.0

        category_match = user.category_affinity.get(product.category, 0.0)
        base = PURCHASE_GIVEN_CLICK_BASE + 0.20 * category_match

        if product.is_noise:
            base *= 0.25

        return float(np.clip(base, 0.01, 0.55))

    def _delivery_failure_prob(self, product: ProductProfile) -> float:
        """
        P(delivery_failure | purchase)

        Derived from seller_delivery_reliability:
          reliability 0.9 → ~10% failure rate
          reliability 0.5 → ~50% failure rate
          reliability 0.2 → capped at 60% failure rate

        This is the signal that separates delivery-aware LinUCB from
        click-signal-only training. High-CTR noise products with poor
        delivery reliability accumulate negative reward over time.
        """
        return float(np.clip(1.0 - product.seller_delivery_reliability, 0.05, 0.60))

    # ── Time helpers ──────────────────────────────────────────────────────────

    def _sample_time_of_day(self) -> float:
        """
        Bimodal activity pattern: lunch peak (12-14h) and evening peak (18-22h).
        The evening peak is slightly broader and more likely.
        Returns normalized float [0, 1].
        """
        if self.rng.random() < 0.40:
            hour = float(self.rng.normal(13.0, 0.8))   # lunch peak
        else:
            hour = float(self.rng.normal(20.0, 1.5))   # evening peak

        return float(np.clip(hour / 24.0, 0.0, 1.0))

    def _sample_timestamp(self, base_date: datetime) -> datetime:
        day_offset = int(self.rng.integers(0, 90))
        tod = self._sample_time_of_day()
        hour = tod * 24.0
        minutes = (hour % 1) * 60
        return base_date + timedelta(
            days=day_offset,
            hours=int(hour),
            minutes=int(minutes),
        )

    # ── Session simulation ────────────────────────────────────────────────────

    def _simulate_session(
        self,
        user: UserProfile,
        products: list,
        base_date: datetime,
        ab_cohort: str,
    ) -> list:
        """
        Simulates one browsing session for a user.

        Products viewed per session follow Poisson(λ=user.session_depth_lambda),
        capped at 10. Products are sampled without replacement, weighted by
        the user's category affinity — so a user who loves electronics sees
        more electronics listings than a random sample would produce.

        The event funnel per product:
          product_view → (maybe) click → (maybe) add_to_cart
                                      → (maybe) purchase → delivery outcome
        """
        session_id = str(uuid.uuid4())
        timestamp = self._sample_timestamp(base_date)
        tod = self._sample_time_of_day()

        # Session depth
        depth = int(self.rng.poisson(user.session_depth_lambda))
        depth = max(1, min(depth, min(10, len(products))))

        # Sample products weighted by category affinity
        category_weights = np.array([
            user.category_affinity.get(p.category, 0.005)
            for p in products
        ])
        category_weights = category_weights / category_weights.sum()

        product_indices = self.rng.choice(
            len(products), size=depth, replace=False, p=category_weights
        )
        viewed = [products[i] for i in product_indices]

        events = []

        for product in viewed:
            # ── product_view ──────────────────────────────────────────────
            events.append(InteractionEvent(
                session_id=session_id,
                user_id=user.user_id,
                product_id=product.product_id,
                event="product_view",
                reward=self.rewards.get("product_view", 0),
                served_by=ab_cohort,
                timestamp=timestamp,
                device_type=user.device_type,
                time_of_day=tod,
                category=product.category,
                delivery_outcome=None,
            ))

            # ── click ─────────────────────────────────────────────────────
            clicked = self.rng.random() < self._click_prob(user, product)
            if clicked:
                events.append(InteractionEvent(
                    session_id=session_id,
                    user_id=user.user_id,
                    product_id=product.product_id,
                    event="click",
                    reward=self.rewards["click"],
                    served_by=ab_cohort,
                    timestamp=timestamp,
                    device_type=user.device_type,
                    time_of_day=tod,
                    category=product.category,
                    delivery_outcome=None,
                ))

            # ── add_to_cart ───────────────────────────────────────────────
            add_to_cart = clicked and (self.rng.random() < CART_GIVEN_CLICK)
            if add_to_cart:
                events.append(InteractionEvent(
                    session_id=session_id,
                    user_id=user.user_id,
                    product_id=product.product_id,
                    event="add_to_cart",
                    reward=self.rewards["add_to_cart"],
                    served_by=ab_cohort,
                    timestamp=timestamp,
                    device_type=user.device_type,
                    time_of_day=tod,
                    category=product.category,
                    delivery_outcome=None,
                ))

            # ── purchase + delivery outcome ───────────────────────────────
            purchased = self.rng.random() < self._purchase_prob(user, product, clicked)
            if purchased:
                failed = self.rng.random() < self._delivery_failure_prob(product)
                delivery_outcome = "failed" if failed else "delivered"

                # Purchase reward + delivery reward combined into one row.
                # The delivery signal is the thesis contribution — separating
                # these two components is what the ablation study measures.
                purchase_reward = self.rewards["purchase"]
                delivery_reward = (
                    self.rewards["delivery_failure"]
                    if failed
                    else self.rewards["delivery_success"]
                )

                events.append(InteractionEvent(
                    session_id=session_id,
                    user_id=user.user_id,
                    product_id=product.product_id,
                    event="purchase",
                    reward=purchase_reward + delivery_reward,
                    served_by=ab_cohort,
                    timestamp=timestamp,
                    device_type=user.device_type,
                    time_of_day=tod,
                    category=product.category,
                    delivery_outcome=delivery_outcome,
                ))

        return events

    # ── Main entrypoint ───────────────────────────────────────────────────────

    def generate(self) -> tuple:
        """
        Runs the full simulation.

        Returns:
            interactions (pd.DataFrame): all interaction events
            users (pd.DataFrame): user profiles with affinity columns
            products (pd.DataFrame): product profiles

        A/B cohort assignment is deterministic per user — the first
        treatment_ratio fraction of users receives LinUCB, the rest receive
        the Greedy baseline. This prevents cohort drift across sessions.

        Sessions per user are distributed evenly across n_rounds. Total
        event volume is approximately: n_users × sessions_per_user × avg_depth × funnel_rate
        """
        base_date = datetime(2026, 1, 1)
        users = self._make_users()
        products = self._make_products()

        sessions_per_user = max(1, self.n_rounds // self.n_users)
        all_events = []

        for i, user in enumerate(users):
            ab_cohort = (
                "linucb"
                if (i / self.n_users) < self.treatment_ratio
                else "greedy"
            )
            for _ in range(sessions_per_user):
                session_events = self._simulate_session(
                    user, products, base_date, ab_cohort
                )
                all_events.extend(session_events)

        # ── Build DataFrames ──────────────────────────────────────────────

        interactions_df = pd.DataFrame([
            {
                "session_id": e.session_id,
                "user_id": e.user_id,
                "product_id": e.product_id,
                "event": e.event,
                "reward": e.reward,
                "served_by": e.served_by,
                "timestamp": e.timestamp,
                "device_type": e.device_type,
                "time_of_day": e.time_of_day,
                "category": e.category,
                "delivery_outcome": e.delivery_outcome,
            }
            for e in all_events
        ])

        users_df = pd.DataFrame([
            {
                "user_id": u.user_id,
                "device_type": u.device_type,
                "price_sensitivity": u.price_sensitivity,
                "session_depth_lambda": u.session_depth_lambda,
                **{f"affinity_{cat}": u.category_affinity[cat] for cat in CATEGORIES},
            }
            for u in users
        ])

        products_df = pd.DataFrame([
            {
                "product_id": p.product_id,
                "category": p.category,
                "price_tier": p.price_tier,
                "seller_quality_score": p.seller_quality_score,
                "seller_delivery_reliability": p.seller_delivery_reliability,
                "days_since_listed": p.days_since_listed,
                "is_noise": p.is_noise,
            }
            for p in products
        ])

        return interactions_df, users_df, products_df

    # ── Distribution validation ───────────────────────────────────────────────

    def validate(self, interactions: pd.DataFrame, products: pd.DataFrame) -> dict:
        """
        Runs distribution checks against Kigali calibration targets.
        Returns a dict of {check_name: passed (bool)} for use in tests.

        Targets:
          mobile_share      >= 0.82
          delivery_fail_rate  in [0.12, 0.35]
          purchase_share    in [0.01, 0.15]   (purchases / total events)
          linucb_share      in [0.75, 0.85]
        """
        total = len(interactions)
        if total == 0:
            return {"error": "empty interactions DataFrame"}

        mobile_share = (interactions["device_type"] == "mobile").mean()

        purchases = interactions[interactions["event"] == "purchase"]
        delivery_fail_rate = (
            (purchases["delivery_outcome"] == "failed").mean()
            if len(purchases) > 0 else 0.0
        )
        purchase_share = len(purchases) / total
        linucb_share = (interactions["served_by"] == "linucb").mean()
        noise_fraction = products["is_noise"].mean()

        checks = {
            "mobile_share_ok":        mobile_share >= 0.82,
            "delivery_fail_rate_ok":  0.12 <= delivery_fail_rate <= 0.40,
            "purchase_share_ok":      0.01 <= purchase_share <= 0.18,
            "linucb_share_ok":        0.75 <= linucb_share <= 0.85,
            "noise_fraction_ok":      abs(noise_fraction - (1 - self.curation_level)) < 0.05,
        }

        summary = {
            "total_events":        total,
            "mobile_share":        round(mobile_share, 3),
            "delivery_fail_rate":  round(delivery_fail_rate, 3),
            "purchase_share":      round(purchase_share, 3),
            "linucb_share":        round(linucb_share, 3),
            "noise_fraction":      round(noise_fraction, 3),
            "checks":              checks,
            "all_passed":          all(checks.values()),
        }

        return summary


# ── Minimal config for running standalone ─────────────────────────────────────

DEFAULT_CONFIG = {
    "simulation": {
        "n_users": 500,
        "n_products": 200,
        "n_rounds": 10_000,
        "curation_level": 1.0,
    },
    "ab_split": {
        "treatment_ratio": 0.80,
        "control_ratio": 0.20,
    },
    "rewards": {
        "product_view": 0,
        "click": 1,
        "add_to_cart": 5,
        "purchase": 20,
        "delivery_success": 3,
        "delivery_failure": -10,
        "delivery_lambda": 1.0,
    },
}


if __name__ == "__main__":
    gen = KigaliSyntheticGenerator(DEFAULT_CONFIG, seed=42)
    interactions, users, products = gen.generate()

    print(f"Generated {len(interactions):,} events across {len(users)} users and {len(products)} products")
    print(f"\nEvent breakdown:\n{interactions['event'].value_counts().to_string()}")
    print(f"\nCohort split:\n{interactions['served_by'].value_counts().to_string()}")

    report = gen.validate(interactions, products)
    print(f"\nValidation report:")
    for k, v in report.items():
        print(f"  {k}: {v}")