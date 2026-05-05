"""
context_builder.py — GiraXpress ml-service version

Assembles the 18-feature context vector for LinUCB.
Unknown product categories fall back to 'accessories' instead of raising —
this makes the service resilient to new GiraXpress category slugs.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from .normalizer import MinMaxNormalizer

# ── Model categories (fixed — changing these changes N_FEATURES) ──────────────

CATEGORIES    = ["electronics", "accessories", "clothing", "home", "beauty"]
N_CATEGORIES  = len(CATEGORIES)
CATEGORY_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}

N_FEATURES = 18  # LinUCB must be initialized with n_features=18

# ── GiraXpress category slug → model category mapping ─────────────────────────
# GiraXpress DB slugs that don't match CATEGORIES are mapped here.
# Fallback for anything not listed: CATEGORIES[1] = 'accessories'

CATEGORY_MAP: dict[str, str] = {
    "electronics":  "electronics",
    "accessories":  "accessories",
    "clothing":     "clothing",
    "home":         "home",
    "beauty":       "beauty",
    # GiraXpress-specific
    "fashion":      "clothing",
    "home-kitchen": "home",
    "food-grocery": "home",
    "sports":       "accessories",
    "books":        "accessories",
    "other":        "accessories",
}

def map_category(slug: str) -> str:
    """Maps any category slug to a valid model category. Never raises."""
    return CATEGORY_MAP.get(slug, "accessories")


# ── Normalizer setup ──────────────────────────────────────────────────────────

_FEATURE_RANGES = {
    "time_of_day":                 (0.0, 1.0),
    "price_tier":                  (0.0, 1.0),
    "seller_quality_score":        (0.0, 1.0),
    "days_since_listed":           (0.0, 1.0),
    "seller_delivery_reliability": (0.0, 1.0),
    "session_depth":               (0.0, 10.0),
}

_NORMALIZER      = MinMaxNormalizer(_FEATURE_RANGES)
_UNIFORM_AFFINITY = np.full(N_CATEGORIES, 1.0 / N_CATEGORIES, dtype=np.float64)


# ── Context builder ───────────────────────────────────────────────────────────

class ContextBuilder:

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
        Returns a normalized context vector of shape (18,).
        Unknown product_category values are mapped via CATEGORY_MAP — never raises.
        """
        # Map category — handles GiraXpress slugs like 'home-kitchen', 'books', etc.
        resolved_category = map_category(product_category)

        vec = np.empty(N_FEATURES, dtype=np.float64)

        # [0] time_of_day
        hour_normalized = timestamp.hour / 24.0 + timestamp.minute / 1440.0
        vec[0] = _NORMALIZER.transform("time_of_day", hour_normalized)

        # [1:3] device one-hot
        vec[1] = 1.0 if device_type == "mobile"  else 0.0
        vec[2] = 1.0 if device_type == "desktop" else 0.0

        # [3:8] category affinity
        if not category_affinity:
            affinity_vec = _UNIFORM_AFFINITY.copy()
        else:
            affinity_vec = np.array(
                [category_affinity.get(cat, 0.0) for cat in CATEGORIES],
                dtype=np.float64,
            )
            total = affinity_vec.sum()
            affinity_vec = affinity_vec / total if total > 0 else _UNIFORM_AFFINITY.copy()
        vec[3:8] = affinity_vec

        # [8] session_depth
        vec[8] = _NORMALIZER.transform("session_depth", float(session_depth))

        # [9] price_tier
        vec[9] = _NORMALIZER.transform("price_tier", price_tier)

        # [10:15] product category one-hot
        cat_vec = np.zeros(N_CATEGORIES, dtype=np.float64)
        cat_vec[CATEGORY_INDEX[resolved_category]] = 1.0
        vec[10:15] = cat_vec

        # [15] seller_quality_score
        vec[15] = _NORMALIZER.transform("seller_quality_score", seller_quality_score)

        # [16] days_since_listed
        vec[16] = _NORMALIZER.transform("days_since_listed", days_since_listed)

        # [17] seller_delivery_reliability
        vec[17] = _NORMALIZER.transform("seller_delivery_reliability", seller_delivery_reliability)

        return vec

    @staticmethod
    def from_synthetic_row(user_row: dict, product_row: dict, timestamp: datetime) -> np.ndarray:
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
        checks = {
            "correct_shape":    vec.shape == (N_FEATURES,),
            "all_in_0_1":       bool(np.all((vec >= 0.0) & (vec <= 1.0))),
            "no_nans":          bool(not np.any(np.isnan(vec))),
            "no_infs":          bool(not np.any(np.isinf(vec))),
            "device_one_hot":   bool(vec[1] + vec[2] == 1.0),
            "affinity_sums_1":  bool(np.isclose(vec[3:8].sum(), 1.0, atol=1e-6)),
            "category_one_hot": bool(vec[10:15].sum() == 1.0),
        }
        checks["all_passed"] = all(checks.values())
        return checks


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