"""
test_bandits.py

Unit tests for LinUCB and GreedyBaseline.

Run:
    pytest tests/test_bandits.py -v

What is tested:
  LinUCB
    - New arm initialized correctly (A=identity, b=zeros)
    - Exploration bonus shrinks as arm is observed more
    - Arm selection is deterministic given fixed context and state
    - Model update is correct (A grows, b grows, theta shifts toward reward)
    - Session buffer accumulates and flushes cleanly
    - Flushing an empty buffer returns 0
    - Attempting to save with a non-empty buffer raises RuntimeError
    - save() / load() round-trip preserves full state
    - Wrong context dimension raises ValueError

  GreedyBaseline
    - Unseen arms score 0.0
    - Mean reward computed correctly after multiple updates
    - Arms ranked by mean reward descending
    - flush() is a no-op (returns 0)
    - save() / load() round-trip preserves full state

  Shared contract
    - Both satisfy the BaseBandit interface
    - rank() returns results sorted descending by score
    - rank() handles a single candidate arm without error
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Allow running from repo root: pytest tests/test_bandits.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bandits.linucb import LinUCB
from bandits.greedy import GreedyBaseline
from bandits.base import BaseBandit


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_FEATURES = 8
ALPHA = 1.0


@pytest.fixture
def linucb():
    return LinUCB(n_features=N_FEATURES, alpha=ALPHA)


@pytest.fixture
def greedy():
    return GreedyBaseline()


@pytest.fixture
def context():
    rng = np.random.default_rng(0)
    return rng.random(N_FEATURES)


@pytest.fixture
def arm_ids():
    return ["arm_a", "arm_b", "arm_c"]


# ── LinUCB: initialization ────────────────────────────────────────────────────

class TestLinUCBInit:

    def test_new_arm_A_is_identity(self, linucb):
        arm = linucb._get_arm("new_arm")
        np.testing.assert_array_equal(arm["A"], np.eye(N_FEATURES))

    def test_new_arm_b_is_zeros(self, linucb):
        arm = linucb._get_arm("new_arm")
        np.testing.assert_array_equal(arm["b"], np.zeros(N_FEATURES))

    def test_arm_count_starts_at_zero(self, linucb):
        assert linucb.arm_count() == 0

    def test_buffer_starts_empty(self, linucb):
        assert linucb.buffer_size() == 0

    def test_total_interactions_starts_at_zero(self, linucb):
        assert linucb.total_interactions == 0


# ── LinUCB: scoring and exploration ──────────────────────────────────────────

class TestLinUCBScoring:

    def test_score_is_float(self, linucb, context):
        s = linucb.score("arm_x", context)
        assert isinstance(s, float)

    def test_score_is_deterministic(self, linucb, context):
        s1 = linucb.score("arm_x", context)
        s2 = linucb.score("arm_x", context)
        assert s1 == s2

    def test_exploration_bonus_shrinks_with_observations(self, linucb, context):
        arm_id = "arm_x"
        bonus_before = linucb.exploration_bonus(arm_id, context)

        # Log 20 interactions with positive reward and flush
        for _ in range(20):
            linucb.log(arm_id, context, reward=1.0)
        linucb.flush()

        bonus_after = linucb.exploration_bonus(arm_id, context)
        assert bonus_after < bonus_before, (
            f"Exploration bonus should shrink after observations. "
            f"Before: {bonus_before:.4f}, After: {bonus_after:.4f}"
        )

    def test_wrong_context_dimension_raises(self, linucb):
        bad_context = np.ones(N_FEATURES + 5)
        with pytest.raises(ValueError, match="features"):
            linucb.rank(["arm_a"], bad_context)

    def test_rank_sorted_descending(self, linucb, arm_ids, context):
        result = linucb.rank(arm_ids, context)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_returns_all_candidates(self, linucb, arm_ids, context):
        result = linucb.rank(arm_ids, context)
        returned_ids = [arm_id for arm_id, _ in result]
        assert set(returned_ids) == set(arm_ids)

    def test_rank_single_arm(self, linucb, context):
        result = linucb.rank(["only_arm"], context)
        assert len(result) == 1
        assert result[0][0] == "only_arm"


# ── LinUCB: updates ───────────────────────────────────────────────────────────

class TestLinUCBUpdate:

    def test_flush_empty_buffer_returns_zero(self, linucb):
        assert linucb.flush() == 0

    def test_flush_returns_count_of_flushed_interactions(self, linucb, context):
        linucb.log("arm_a", context, reward=1.0)
        linucb.log("arm_b", context, reward=5.0)
        linucb.log("arm_a", context, reward=20.0)
        n = linucb.flush()
        assert n == 3

    def test_buffer_cleared_after_flush(self, linucb, context):
        linucb.log("arm_a", context, reward=1.0)
        linucb.flush()
        assert linucb.buffer_size() == 0

    def test_total_interactions_increments_on_flush(self, linucb, context):
        linucb.log("arm_a", context, reward=1.0)
        linucb.log("arm_a", context, reward=1.0)
        linucb.flush()
        assert linucb.total_interactions == 2

    def test_A_matrix_grows_after_update(self, linucb, context):
        A_before = linucb._get_arm("arm_x")["A"].copy()
        linucb.log("arm_x", context, reward=1.0)
        linucb.flush()
        A_after = linucb._arms["arm_x"]["A"]
        # A should have increased: A_new = A_old + x x^T
        assert not np.allclose(A_before, A_after)

    def test_A_update_matches_outer_product(self, linucb, context):
        linucb._get_arm("arm_x")  # initialize
        A_before = linucb._arms["arm_x"]["A"].copy()
        linucb.log("arm_x", context, reward=1.0)
        linucb.flush()
        A_after = linucb._arms["arm_x"]["A"]
        expected = A_before + np.outer(context, context)
        np.testing.assert_allclose(A_after, expected)

    def test_b_update_matches_reward_times_context(self, linucb, context):
        linucb._get_arm("arm_x")  # initialize
        reward = 5.0
        linucb.log("arm_x", context, reward=reward)
        linucb.flush()
        expected_b = reward * context  # b starts at zeros
        np.testing.assert_allclose(linucb._arms["arm_x"]["b"], expected_b)

    def test_theta_shifts_toward_high_reward_direction(self):
        """
        After many observations with reward = context[0] (i.e., reward is
        perfectly predicted by the first feature), theta[0] should be
        the largest weight.
        """
        rng = np.random.default_rng(7)
        model = LinUCB(n_features=5, alpha=0.01)  # low alpha = mostly exploit

        for _ in range(200):
            ctx = rng.random(5)
            reward = ctx[0] * 20  # reward is driven entirely by feature 0
            model.log("arm_x", ctx, reward=reward)
        model.flush()

        theta = model.theta("arm_x")
        assert np.argmax(theta) == 0, (
            f"Expected feature 0 to have highest weight. Got theta={theta}"
        )

    def test_high_reward_arm_ranked_first_after_many_observations(self):
        """
        After repeated observations, the arm with consistently higher reward
        should rank above the arm with consistently lower reward.
        """
        rng = np.random.default_rng(99)
        model = LinUCB(n_features=4, alpha=0.1)

        ctx = rng.random(4)

        for _ in range(50):
            model.log("good_arm", ctx, reward=20.0)
            model.log("bad_arm", ctx, reward=0.0)
        model.flush()

        result = model.rank(["good_arm", "bad_arm"], ctx)
        assert result[0][0] == "good_arm", (
            f"good_arm should rank first. Got: {result}"
        )


# ── LinUCB: persistence ───────────────────────────────────────────────────────

class TestLinUCBPersistence:

    def test_save_with_nonempty_buffer_raises(self, linucb, context):
        linucb.log("arm_x", context, reward=1.0)
        with pytest.raises(RuntimeError, match="[Ff]lush"):
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                linucb.save(f.name)

    def test_save_load_roundtrip_preserves_arm_count(self, linucb, context):
        for arm_id in ["arm_a", "arm_b", "arm_c"]:
            linucb.log(arm_id, context, reward=1.0)
        linucb.flush()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            linucb.save(path)
            loaded = LinUCB.load(path)
            assert loaded.arm_count() == linucb.arm_count()
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_preserves_A_matrices(self, linucb, context):
        arm_id = "arm_x"
        linucb.log(arm_id, context, reward=10.0)
        linucb.flush()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            linucb.save(path)
            loaded = LinUCB.load(path)
            np.testing.assert_allclose(
                loaded._arms[arm_id]["A"],
                linucb._arms[arm_id]["A"],
            )
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_preserves_alpha(self, linucb):
        linucb.flush()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            linucb.save(path)
            loaded = LinUCB.load(path)
            assert loaded.alpha == linucb.alpha
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_preserves_scores(self, linucb, context, arm_ids):
        for arm_id in arm_ids:
            linucb.log(arm_id, context, reward=float(len(arm_id)))
        linucb.flush()

        original_scores = {
            arm_id: linucb.score(arm_id, context) for arm_id in arm_ids
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            linucb.save(path)
            loaded = LinUCB.load(path)
            for arm_id in arm_ids:
                np.testing.assert_allclose(
                    loaded.score(arm_id, context),
                    original_scores[arm_id],
                    rtol=1e-10,
                )
        finally:
            os.unlink(path)


# ── GreedyBaseline: scoring ───────────────────────────────────────────────────

class TestGreedyScoring:

    def test_unseen_arm_scores_zero(self, greedy, context):
        result = greedy.rank(["unseen_arm"], context)
        assert result[0][1] == 0.0

    def test_mean_reward_correct(self, greedy):
        greedy.log("arm_a", reward=10.0)
        greedy.log("arm_a", reward=20.0)
        assert greedy._mean_reward("arm_a") == pytest.approx(15.0)

    def test_rank_sorted_descending(self, greedy):
        greedy.log("arm_low", reward=1.0)
        greedy.log("arm_high", reward=20.0)
        greedy.log("arm_mid", reward=5.0)
        result = greedy.rank(["arm_low", "arm_high", "arm_mid"])
        ids = [arm_id for arm_id, _ in result]
        assert ids == ["arm_high", "arm_mid", "arm_low"]

    def test_flush_is_noop(self, greedy):
        greedy.log("arm_a", reward=1.0)
        n = greedy.flush()
        assert n == 0

    def test_total_interactions_increments_on_log(self, greedy):
        greedy.log("arm_a", reward=1.0)
        greedy.log("arm_b", reward=5.0)
        assert greedy.total_interactions == 2

    def test_rank_single_arm(self, greedy, context):
        result = greedy.rank(["only_arm"], context)
        assert len(result) == 1

    def test_context_ignored(self, greedy):
        greedy.log("arm_x", reward=10.0)
        ctx_a = np.ones(5)
        ctx_b = np.zeros(5)
        score_a = greedy.rank(["arm_x"], ctx_a)[0][1]
        score_b = greedy.rank(["arm_x"], ctx_b)[0][1]
        assert score_a == score_b


# ── GreedyBaseline: persistence ───────────────────────────────────────────────

class TestGreedyPersistence:

    def test_save_load_roundtrip_preserves_mean_reward(self, greedy):
        greedy.log("arm_a", reward=10.0)
        greedy.log("arm_a", reward=30.0)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            greedy.save(path)
            loaded = GreedyBaseline.load(path)
            assert loaded._mean_reward("arm_a") == pytest.approx(20.0)
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_preserves_total_interactions(self, greedy):
        greedy.log("arm_a", reward=1.0)
        greedy.log("arm_b", reward=1.0)
        greedy.log("arm_c", reward=1.0)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            greedy.save(path)
            loaded = GreedyBaseline.load(path)
            assert loaded.total_interactions == 3
        finally:
            os.unlink(path)


# ── Shared contract ───────────────────────────────────────────────────────────

class TestSharedContract:

    def test_linucb_is_base_bandit(self, linucb):
        assert isinstance(linucb, BaseBandit)

    def test_greedy_is_base_bandit(self, greedy):
        assert isinstance(greedy, BaseBandit)

    @pytest.mark.parametrize("bandit_fixture", ["linucb", "greedy"])
    def test_rank_returns_list(self, bandit_fixture, request, context, arm_ids):
        bandit = request.getfixturevalue(bandit_fixture)
        result = bandit.rank(arm_ids, context)
        assert isinstance(result, list)

    @pytest.mark.parametrize("bandit_fixture", ["linucb", "greedy"])
    def test_rank_length_matches_candidates(self, bandit_fixture, request, context, arm_ids):
        bandit = request.getfixturevalue(bandit_fixture)
        result = bandit.rank(arm_ids, context)
        assert len(result) == len(arm_ids)

    @pytest.mark.parametrize("bandit_fixture", ["linucb", "greedy"])
    def test_rank_tuples_have_two_elements(self, bandit_fixture, request, context, arm_ids):
        bandit = request.getfixturevalue(bandit_fixture)
        result = bandit.rank(arm_ids, context)
        for item in result:
            assert len(item) == 2

    @pytest.mark.parametrize("bandit_fixture", ["linucb", "greedy"])
    def test_scores_are_floats(self, bandit_fixture, request, context, arm_ids):
        bandit = request.getfixturevalue(bandit_fixture)
        result = bandit.rank(arm_ids, context)
        for _, score in result:
            assert isinstance(score, float)