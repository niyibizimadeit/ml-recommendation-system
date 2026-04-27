"""
test_features.py

Unit tests for context_builder.py and normalizer.py.

Run:
    pytest tests/test_features.py -v

What is tested:

  ContextBuilder
    - Output shape is (18,) = N_FEATURES
    - All values in [0, 1]
    - No NaNs or Infs
    - Device one-hot sums to 1.0
    - Category affinity sums to 1.0
    - Product category one-hot sums to 1.0
    - Unknown user (None affinity) gets uniform prior
    - Zero-sum affinity dict falls back to uniform prior
    - Invalid product category raises ValueError
    - time_of_day maps correctly (midnight → 0.0, noon → 0.5)
    - Mobile vs desktop produces different device slice
    - Delivery reliability index is 17
    - from_synthetic_row produces valid vector
    - Vectors for different products differ
    - Vectors for same inputs are identical (deterministic)

  MinMaxNormalizer
    - Known value maps correctly
    - Value below min clips to 0.0
    - Value above max clips to 1.0
    - Unknown feature raises KeyError
    - Equal min/max returns 0.0 (no division by zero)
    - transform_array produces correct shape and values

  ZScoreNormalizer
    - Before fit returns 0.5
    - After fit, mean maps to 0.5 (sigmoid of 0)
    - Values above mean map above 0.5
    - Values below mean map below 0.5
    - Output always in [0, 1]
    - Save/load round-trip preserves mean and std
"""

import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.context_builder import (
    ContextBuilder,
    build_context,
    N_FEATURES,
    CATEGORIES,
)
from features.normalizer import MinMaxNormalizer, ZScoreNormalizer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_kwargs():
    """Minimal valid kwargs for ContextBuilder.build()."""
    return dict(
        timestamp=datetime(2026, 3, 15, 14, 30),
        device_type="mobile",
        category_affinity={"electronics": 0.6, "accessories": 0.4},
        session_depth=3,
        price_tier=0.4,
        product_category="electronics",
        seller_quality_score=0.85,
        days_since_listed=0.1,
        seller_delivery_reliability=0.9,
    )


@pytest.fixture
def built_vector(base_kwargs):
    return ContextBuilder.build(**base_kwargs)


# ── ContextBuilder: output shape and values ───────────────────────────────────

class TestContextBuilderShape:

    def test_output_shape(self, built_vector):
        assert built_vector.shape == (N_FEATURES,)

    def test_all_values_in_0_1(self, built_vector):
        assert np.all((built_vector >= 0.0) & (built_vector <= 1.0)), (
            f"Values out of [0,1]: {built_vector}"
        )

    def test_no_nans(self, built_vector):
        assert not np.any(np.isnan(built_vector))

    def test_no_infs(self, built_vector):
        assert not np.any(np.isinf(built_vector))

    def test_validate_passes_on_valid_vector(self, built_vector):
        report = ContextBuilder.validate_vector(built_vector)
        assert report["all_passed"], f"Validation failed: {report}"


class TestContextBuilderSlices:

    def test_device_mobile_slice(self, base_kwargs):
        base_kwargs["device_type"] = "mobile"
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[1] == 1.0   # mobile
        assert vec[2] == 0.0   # desktop

    def test_device_desktop_slice(self, base_kwargs):
        base_kwargs["device_type"] = "desktop"
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[1] == 0.0   # mobile
        assert vec[2] == 1.0   # desktop

    def test_device_one_hot_sums_to_one(self, built_vector):
        assert built_vector[1] + built_vector[2] == 1.0

    def test_affinity_slice_sums_to_one(self, built_vector):
        assert np.isclose(built_vector[3:8].sum(), 1.0, atol=1e-6)

    def test_category_one_hot_sums_to_one(self, built_vector):
        assert built_vector[10:15].sum() == 1.0

    def test_correct_category_bit_set(self, base_kwargs):
        base_kwargs["product_category"] = "clothing"
        vec = ContextBuilder.build(**base_kwargs)
        # clothing is index 2 in CATEGORIES → offset 10+2 = 12
        assert vec[12] == 1.0
        # all others in the category slice should be 0
        cat_slice = vec[10:15].tolist()
        cat_slice.pop(2)
        assert all(v == 0.0 for v in cat_slice)

    def test_delivery_reliability_is_index_17(self, base_kwargs):
        base_kwargs["seller_delivery_reliability"] = 0.0
        vec_low = ContextBuilder.build(**base_kwargs)
        base_kwargs["seller_delivery_reliability"] = 1.0
        vec_high = ContextBuilder.build(**base_kwargs)
        assert vec_low[17] < vec_high[17]

    def test_seller_quality_is_index_15(self, base_kwargs):
        base_kwargs["seller_quality_score"] = 0.0
        vec_low = ContextBuilder.build(**base_kwargs)
        base_kwargs["seller_quality_score"] = 1.0
        vec_high = ContextBuilder.build(**base_kwargs)
        assert vec_low[15] < vec_high[15]


class TestContextBuilderTime:

    def test_midnight_maps_to_zero(self, base_kwargs):
        base_kwargs["timestamp"] = datetime(2026, 1, 1, 0, 0)
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[0] == pytest.approx(0.0, abs=1e-3)

    def test_noon_maps_to_half(self, base_kwargs):
        base_kwargs["timestamp"] = datetime(2026, 1, 1, 12, 0)
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[0] == pytest.approx(0.5, abs=1e-3)

    def test_11pm_maps_near_one(self, base_kwargs):
        base_kwargs["timestamp"] = datetime(2026, 1, 1, 23, 0)
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[0] > 0.9


# ── ContextBuilder: cold-start and edge cases ─────────────────────────────────

class TestContextBuilderColdStart:

    def test_none_affinity_gives_uniform(self, base_kwargs):
        base_kwargs["category_affinity"] = None
        vec = ContextBuilder.build(**base_kwargs)
        expected = np.full(5, 1.0 / 5)
        np.testing.assert_allclose(vec[3:8], expected, atol=1e-6)

    def test_empty_affinity_gives_uniform(self, base_kwargs):
        base_kwargs["category_affinity"] = {}
        vec = ContextBuilder.build(**base_kwargs)
        expected = np.full(5, 1.0 / 5)
        np.testing.assert_allclose(vec[3:8], expected, atol=1e-6)

    def test_zero_sum_affinity_gives_uniform(self, base_kwargs):
        base_kwargs["category_affinity"] = {cat: 0.0 for cat in CATEGORIES}
        vec = ContextBuilder.build(**base_kwargs)
        expected = np.full(5, 1.0 / 5)
        np.testing.assert_allclose(vec[3:8], expected, atol=1e-6)

    def test_invalid_category_raises_value_error(self, base_kwargs):
        base_kwargs["product_category"] = "furniture"
        with pytest.raises(ValueError, match="Unknown product category"):
            ContextBuilder.build(**base_kwargs)

    def test_session_depth_zero_maps_to_zero(self, base_kwargs):
        base_kwargs["session_depth"] = 0
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[8] == 0.0

    def test_session_depth_ten_maps_to_one(self, base_kwargs):
        base_kwargs["session_depth"] = 10
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[8] == pytest.approx(1.0)

    def test_session_depth_above_cap_clips(self, base_kwargs):
        base_kwargs["session_depth"] = 999
        vec = ContextBuilder.build(**base_kwargs)
        assert vec[8] == pytest.approx(1.0)


# ── ContextBuilder: determinism and differentiation ──────────────────────────

class TestContextBuilderDeterminism:

    def test_same_inputs_produce_same_vector(self, base_kwargs):
        v1 = ContextBuilder.build(**base_kwargs)
        v2 = ContextBuilder.build(**base_kwargs)
        np.testing.assert_array_equal(v1, v2)

    def test_different_products_produce_different_vectors(self, base_kwargs):
        v1 = ContextBuilder.build(**base_kwargs)
        base_kwargs["product_category"] = "beauty"
        base_kwargs["price_tier"] = 0.9
        v2 = ContextBuilder.build(**base_kwargs)
        assert not np.array_equal(v1, v2)

    def test_module_level_function_matches_class(self, base_kwargs):
        v_class = ContextBuilder.build(**base_kwargs)
        v_func = build_context(**base_kwargs)
        np.testing.assert_array_equal(v_class, v_func)


# ── ContextBuilder: from_synthetic_row ───────────────────────────────────────

class TestFromSyntheticRow:

    def test_produces_valid_vector(self):
        user_row = {
            "device_type": "mobile",
            "session_depth": 2,
            "affinity_electronics": 0.5,
            "affinity_accessories": 0.2,
            "affinity_clothing": 0.1,
            "affinity_home": 0.1,
            "affinity_beauty": 0.1,
        }
        product_row = {
            "category": "electronics",
            "price_tier": 0.4,
            "seller_quality_score": 0.8,
            "days_since_listed": 0.2,
            "seller_delivery_reliability": 0.9,
        }
        ts = datetime(2026, 3, 15, 20, 0)
        vec = ContextBuilder.from_synthetic_row(user_row, product_row, ts)
        report = ContextBuilder.validate_vector(vec)
        assert report["all_passed"], f"Validation failed: {report}"

    def test_missing_affinity_keys_use_uniform(self):
        user_row = {"device_type": "desktop", "session_depth": 1}
        product_row = {
            "category": "clothing",
            "price_tier": 0.3,
            "seller_quality_score": 0.7,
            "days_since_listed": 0.5,
            "seller_delivery_reliability": 0.6,
        }
        vec = ContextBuilder.from_synthetic_row(user_row, product_row, datetime.now())
        # Affinity slice should sum to 1.0 even with missing keys
        assert np.isclose(vec[3:8].sum(), 1.0, atol=1e-6)


# ── MinMaxNormalizer ──────────────────────────────────────────────────────────

class TestMinMaxNormalizer:

    @pytest.fixture
    def norm(self):
        return MinMaxNormalizer({"price": (0.0, 10_000.0), "age": (0.0, 90.0)})

    def test_known_value_maps_correctly(self, norm):
        assert norm.transform("price", 5_000.0) == pytest.approx(0.5)

    def test_min_maps_to_zero(self, norm):
        assert norm.transform("price", 0.0) == pytest.approx(0.0)

    def test_max_maps_to_one(self, norm):
        assert norm.transform("price", 10_000.0) == pytest.approx(1.0)

    def test_below_min_clips_to_zero(self, norm):
        assert norm.transform("price", -500.0) == pytest.approx(0.0)

    def test_above_max_clips_to_one(self, norm):
        assert norm.transform("price", 50_000.0) == pytest.approx(1.0)

    def test_unknown_feature_raises_key_error(self, norm):
        with pytest.raises(KeyError):
            norm.transform("unknown_feature", 1.0)

    def test_equal_min_max_returns_zero(self):
        norm = MinMaxNormalizer({"flat": (5.0, 5.0)})
        assert norm.transform("flat", 5.0) == 0.0

    def test_transform_array_correct_shape(self, norm):
        values = np.array([0.0, 2500.0, 5000.0, 10000.0])
        result = norm.transform_array("price", values)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, [0.0, 0.25, 0.5, 1.0])


# ── ZScoreNormalizer ──────────────────────────────────────────────────────────

class TestZScoreNormalizer:

    @pytest.fixture
    def fitted_norm(self):
        norm = ZScoreNormalizer()
        norm.fit(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        return norm

    def test_before_fit_returns_half(self):
        norm = ZScoreNormalizer()
        assert norm.transform(999.0) == pytest.approx(0.5)

    def test_mean_maps_to_half(self, fitted_norm):
        # sigmoid(0) = 0.5, so the mean should map to 0.5
        mean = 30.0
        assert fitted_norm.transform(mean) == pytest.approx(0.5, abs=1e-6)

    def test_above_mean_maps_above_half(self, fitted_norm):
        assert fitted_norm.transform(50.0) > 0.5

    def test_below_mean_maps_below_half(self, fitted_norm):
        assert fitted_norm.transform(10.0) < 0.5

    def test_output_always_in_0_1(self, fitted_norm):
        for v in [-1000.0, 0.0, 30.0, 1000.0]:
            result = fitted_norm.transform(v)
            assert 0.0 <= result <= 1.0

    def test_save_load_roundtrip(self, fitted_norm):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_norm.save(path)
            loaded = ZScoreNormalizer.load(path)
            assert loaded._mean == fitted_norm._mean
            assert loaded._std == fitted_norm._std
            assert loaded._fitted == True
        finally:
            os.unlink(path)

    def test_transform_array_shape(self, fitted_norm):
        arr = np.array([10.0, 20.0, 30.0])
        result = fitted_norm.transform_array(arr)
        assert result.shape == (3,)

    def test_transform_array_all_in_0_1(self, fitted_norm):
        arr = np.linspace(-100, 100, 50)
        result = fitted_norm.transform_array(arr)
        assert np.all((result >= 0.0) & (result <= 1.0))