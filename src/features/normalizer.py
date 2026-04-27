"""
normalizer.py

Min-max and z-score normalization for context vector features.

All continuous features must land in [0, 1] before being passed to LinUCB.
Raw values fed into the dot product let high-magnitude features (price in RWF,
days since listed) dominate the learned weights. Normalization prevents this.

Two strategies:

  MinMaxNormalizer — clips to a known [min, max] range defined per feature.
    Use when the range is known and stable (time of day, price tier, days
    since listed). Values outside the range are clipped, not errored.

  ZScoreNormalizer — standardizes using a running mean and std.
    Use when the range shifts over time (category affinity as the catalog
    grows, seller quality scores after bulk re-scoring). Requires fitting
    on a sample before use.

The context_builder uses MinMaxNormalizer for all features in Phase ML-3
because the ranges are defined and stable. ZScoreNormalizer is provided
for future use in Phase ML-6 (ablation studies with shifted distributions).
"""

import pickle
import numpy as np
from pathlib import Path


class MinMaxNormalizer:
    """
    Clips and scales a value to [0, 1] given a known [min, max] range.

    Usage:
        norm = MinMaxNormalizer(feature_ranges={
            "time_of_day": (0.0, 1.0),
            "price_tier":  (0.0, 1.0),
        })
        scaled = norm.transform("time_of_day", 0.75)  # → 0.75

    Values outside the range are clipped silently. This handles edge cases
    like a product listed 95 days ago when the cap is 90 days.
    """

    def __init__(self, feature_ranges: dict):
        """
        Args:
            feature_ranges: {feature_name: (min_val, max_val)}
        """
        self.feature_ranges = feature_ranges

    def transform(self, feature_name: str, value: float) -> float:
        if feature_name not in self.feature_ranges:
            raise KeyError(
                f"Feature '{feature_name}' not in normalizer. "
                f"Known features: {list(self.feature_ranges.keys())}"
            )
        lo, hi = self.feature_ranges[feature_name]
        if hi == lo:
            return 0.0
        clipped = np.clip(value, lo, hi)
        return float((clipped - lo) / (hi - lo))

    def transform_array(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        lo, hi = self.feature_ranges[feature_name]
        if hi == lo:
            return np.zeros_like(values, dtype=np.float64)
        clipped = np.clip(values, lo, hi)
        return (clipped - lo) / (hi - lo)


class ZScoreNormalizer:
    """
    Standardizes values using running mean and std: z = (x - μ) / σ
    then clips to [0, 1] via sigmoid to keep output bounded.

    Requires fitting on a sample before use. Safe to call transform()
    before fit() — returns 0.5 (the sigmoid midpoint) as a safe default.

    Usage:
        norm = ZScoreNormalizer()
        norm.fit(sample_values)
        scaled = norm.transform(raw_value)
    """

    def __init__(self):
        self._mean: float = 0.0
        self._std: float = 1.0
        self._fitted: bool = False

    def fit(self, values: np.ndarray) -> "ZScoreNormalizer":
        self._mean = float(np.mean(values))
        self._std = float(np.std(values)) or 1.0
        self._fitted = True
        return self

    def transform(self, value: float) -> float:
        if not self._fitted:
            return 0.5
        z = (value - self._mean) / self._std
        return float(1.0 / (1.0 + np.exp(-z)))  # sigmoid to [0, 1]

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.full_like(values, 0.5, dtype=np.float64)
        z = (values - self._mean) / self._std
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mean": self._mean, "std": self._std, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> "ZScoreNormalizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._mean = state["mean"]
        obj._std = state["std"]
        obj._fitted = state["fitted"]
        return obj