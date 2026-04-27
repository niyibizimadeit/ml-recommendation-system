"""
context_builder.py

Assembles the context vector passed to LinUCB on every recommendation call.

The vector is the concatenation of user features and product features.
Every feature is normalized to [0, 1] before assembly.

Final vector shape: 18 features (N_FEATURES = 18)

  Index  Feature                          Type        Source
  ─────────────────────────────────────────────────────────────────────
  0      time_of_day                      float       request timestamp
  1      device_mobile                    binary      user agent / cookie
  2      device_desktop                   binary      user agent / cookie
  3      affinity_electronics             float       interaction history
  4      affinity_accessories             float       interaction history
  5      affinity_clothing                float       interaction history
  6      affinity_home                    float       interaction history
  7      affinity_beauty                  float       interaction history
  8      session_depth                    float       current session
  9      price_tier                       float       product record
  10     category_electronics             binary      product record
  11     category_accessories             binary      product record
  12     category_clothing                binary      product record
  13     category_home                    binary      product record
  14     category_beauty                  binary      product record
  15     seller_quality_score             float       seller record
  16     days_since_listed                float       product record
  17     seller_delivery_reliability      float       seller record (Phase 15)

N_FEATURES = 18. Pass this constant to LinUCB(n_features=18).

The build() method is the only public API. It accepts raw values and
returns a normalized numpy array ready for LinUCB.score() or LinUCB.rank().

Cold-start defaults:
  - Unknown user → all affinity scores = 1/n_categories (uniform)
  - Unknown seller quality → 0.5 (mid-range, conservative)
  - Unknown delivery reliability → 0.5 (mid-range, conservative)
  - Session depth 0 → normalized to 0.0
"""

import numpy as np
from datetime import datetime
from typing import Optional

from .normalizer import MinMaxNormalizer

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = ["electronics", "accessories", "clothing", "home", "beauty"]
N_CATEGORIES = len(CATEGORIES)
CATEGORY_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}

N_FEATURES = 18  # export this — LinUCB must be initialized with this value

_FEATURE_RANGES = {
    "time_of_day":                  (0.0, 1.0),
    "price_tier":                   (0.0, 1.0),
    "seller_quality_score":         (0.0, 1.0),
    "days_since_listed":            (0.0, 1.0),   # raw value already normalized 0-90→0-1
    "seller_delivery_reliability":  (0.0, 1.0),
    "session_depth":                (0.0, 10.0),  # capped at 10 pages per session
}

_NORMALIZER = MinMaxNormalizer(_FEATURE_RANGES)

# Uniform prior for cold-start users: 1/n_categories per category
_UNIFORM_AFFINITY = np.full(N_CATEGORIES, 1.0 / N_CATEGORIES, dtype=np.float64)


# ── Context builder ───────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Stateless context vector assembler.

    All methods are static — no internal state. Instantiate once and reuse,
    or call build() directly as a module-level function via the convenience
    wrapper at the bottom of this file.
    """

    @staticmethod
    def build(
        timestamp: datetime,
        device_type: str,
        category_affinity: Optional[dict],
        session_depth: int,
        price_tier: float,
        product_category: str,
        seller_quality_score: float,
        days_since_listed: float,
        seller_delivery_reliability: float,
    ) -> np.ndarray:
        """
        Assembles and returns a normalized context vector of shape (18,).

        Args:
            timestamp:                    request time (used for time_of_day)
            device_type:                  'mobile' or 'desktop'
            category_affinity:            {category: weight} dict, sums to ~1.0.
                                          Pass None for new/guest users.
            session_depth:                pages viewed in current session (int)
            price_tier:                   product price tier, already [0, 1]
            product_category:             one of CATEGORIES
            seller_quality_score:         [0, 1], default 0.5 if unknown
            days_since_listed:            [0, 1], normalized from 0-90 day raw
            seller_delivery_reliability:  [0, 1], default 0.5 if unknown

        Returns:
            np.ndarray of shape (18,), all values in [0, 1]

        Raises:
            ValueError if product_category is not in CATEGORIES
        """
        if product_category not in CATEGORY_INDEX:
            raise ValueError(
                f"Unknown product category '{product_category}'. "
                f"Valid categories: {CATEGORIES}"
            )

        vec = np.empty(N_FEATURES, dtype=np.float64)

        # ── [0] time_of_day ───────────────────────────────────────────────
        hour_normalized = timestamp.hour / 24.0 + timestamp.minute / 1440.0
        vec[0] = _NORMALIZER.transform("time_of_day", hour_normalized)

        # ── [1:3] device one-hot ──────────────────────────────────────────
        vec[1] = 1.0 if device_type == "mobile" else 0.0
        vec[2] = 1.0 if device_type == "desktop" else 0.0

        # ── [3:8] category affinity ───────────────────────────────────────
        if category_affinity is None or len(category_affinity) == 0:
            affinity_vec = _UNIFORM_AFFINITY.copy()
        else:
            affinity_vec = np.array(
                [category_affinity.get(cat, 0.0) for cat in CATEGORIES],
                dtype=np.float64,
            )
            total = affinity_vec.sum()
            if total > 0:
                affinity_vec /= total  # normalize to sum to 1.0
            else:
                affinity_vec = _UNIFORM_AFFINITY.copy()
        vec[3:8] = affinity_vec

        # ── [8] session_depth ─────────────────────────────────────────────
        vec[8] = _NORMALIZER.transform("session_depth", float(session_depth))

        # ── [9] price_tier ────────────────────────────────────────────────
        vec[9] = _NORMALIZER.transform("price_tier", price_tier)

        # ── [10:15] product category one-hot ──────────────────────────────
        cat_vec = np.zeros(N_CATEGORIES, dtype=np.float64)
        cat_vec[CATEGORY_INDEX[product_category]] = 1.0
        vec[10:15] = cat_vec

        # ── [15] seller_quality_score ─────────────────────────────────────
        vec[15] = _NORMALIZER.transform("seller_quality_score", seller_quality_score)

        # ── [16] days_since_listed ────────────────────────────────────────
        vec[16] = _NORMALIZER.transform("days_since_listed", days_since_listed)

        # ── [17] seller_delivery_reliability ─────────────────────────────
        vec[17] = _NORMALIZER.transform("seller_delivery_reliability", seller_delivery_reliability)

        return vec

    @staticmethod
    def from_synthetic_row(user_row: dict, product_row: dict, timestamp: datetime) -> np.ndarray:
        """
        Convenience builder for simulation notebooks.
        Accepts raw dicts from the synthetic generator DataFrames.

        user_row:    one row from users_df (with affinity_* columns)
        product_row: one row from products_df
        """
        affinity = {
            cat: user_row.get(f"affinity_{cat}", 1.0 / N_CATEGORIES)
            for cat in CATEGORIES
        }
        return ContextBuilder.build(
            timestamp=timestamp,
            device_type=user_row.get("device_type", "mobile"),
            category_affinity=affinity,
            session_depth=user_row.get("session_depth", 0),
            price_tier=product_row.get("price_tier", 0.5),
            product_category=product_row.get("category", CATEGORIES[0]),
            seller_quality_score=product_row.get("seller_quality_score", 0.5),
            days_since_listed=product_row.get("days_since_listed", 0.5),
            seller_delivery_reliability=product_row.get("seller_delivery_reliability", 0.5),
        )

    @staticmethod
    def validate_vector(vec: np.ndarray) -> dict:
        """
        Checks a built context vector for correctness.
        Returns a report dict. Use in tests and notebook sanity checks.
        """
        checks = {
            "correct_shape":   vec.shape == (N_FEATURES,),
            "all_in_0_1":      bool(np.all((vec >= 0.0) & (vec <= 1.0))),
            "no_nans":         bool(not np.any(np.isnan(vec))),
            "no_infs":         bool(not np.any(np.isinf(vec))),
            "device_one_hot":  bool(vec[1] + vec[2] == 1.0),
            "affinity_sums_1": bool(np.isclose(vec[3:8].sum(), 1.0, atol=1e-6)),
            "category_one_hot": bool(vec[10:15].sum() == 1.0),
        }
        checks["all_passed"] = all(checks.values())
        return checks


# ── Module-level convenience function ─────────────────────────────────────────

def build_context(
    timestamp: datetime,
    device_type: str,
    category_affinity: Optional[dict],
    session_depth: int,
    price_tier: float,
    product_category: str,
    seller_quality_score: float,
    days_since_listed: float,
    seller_delivery_reliability: float,
) -> np.ndarray:
    """
    Module-level wrapper around ContextBuilder.build().
    Import this for cleaner call sites in the API and notebooks.

    Example:
        from features.context_builder import build_context, N_FEATURES
        ctx = build_context(...)
    """
    return ContextBuilder.build(
        timestamp=timestamp,
        device_type=device_type,
        category_affinity=category_affinity,
        session_depth=session_depth,
        price_tier=price_tier,
        product_category=product_category,
        seller_quality_score=seller_quality_score,
        days_since_listed=days_since_listed,
        seller_delivery_reliability=seller_delivery_reliability,
    )