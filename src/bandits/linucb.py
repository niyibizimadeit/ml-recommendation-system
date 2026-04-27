"""
linucb.py

LinUCB (Linear Upper Confidence Bound) contextual bandit.

This is the thesis treatment arm. It models expected reward as a linear
function of a context vector and balances exploration vs exploitation via
a confidence bonus derived from the per-arm feature covariance matrix.

Selection rule:
    a_t = argmax_a [ θ_a^T x_t + α √(x_t^T A_a^{-1} x_t) ]

Where:
    x_t   — context vector at round t
    θ_a   — learned weight vector for arm a (= A_a^{-1} b_a)
    A_a   — regularized feature covariance matrix for arm a
    α     — exploration coefficient (calibrated in notebook 02)

Update rule (applied at session flush):
    A_a ← A_a + x_t x_t^T
    b_a ← b_a + r_t · x_t

New arms are initialized with A = I (identity) and b = 0 (zeros).
The identity initialization means every arm starts with equal uncertainty —
LinUCB will explore all arms before exploiting confident ones.

Session-level updates:
    Interactions are buffered via log() and applied in one pass via flush().
    This is cheaper than per-event updates and avoids race conditions in
    a multi-threaded server context. Call flush() once at session end.

Model persistence:
    save() serializes A matrices, b vectors, alpha, and n_features to disk
    using pickle. load() reconstructs the full model. The recommended pattern
    is a periodic cron snapshot — not a save on every flush.
"""

import pickle
import numpy as np
from pathlib import Path

from .base import BaseBandit


class LinUCB(BaseBandit):
    """
    Disjoint LinUCB: each arm maintains independent A and b matrices.

    Args:
        n_features: dimensionality of the context vector. Must match
                    the output of context_builder.py exactly.
        alpha:      exploration coefficient. Higher = more exploration.
                    Calibrated via notebooks/02_alpha_sensitivity.ipynb.
    """

    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha

        # Per-arm state. Keyed by arm_id (product_id string).
        # Each entry: {'A': ndarray (d,d), 'b': ndarray (d,)}
        self._arms: dict = {}

        # Session buffer: list of (arm_id, context, reward) tuples.
        # Cleared after every flush().
        self._buffer: list = []

        # Running interaction count — used for regret tracking.
        self.total_interactions: int = 0

    # ── Arm initialization ────────────────────────────────────────────────────

    def _init_arm(self, arm_id: str) -> dict:
        """
        Creates a fresh arm state. A = identity, b = zeros.

        Identity initialization gives each new arm the same prior uncertainty,
        so LinUCB will explore it before committing to exploitation.
        """
        arm = {
            "A": np.eye(self.n_features, dtype=np.float64),
            "b": np.zeros(self.n_features, dtype=np.float64),
        }
        self._arms[arm_id] = arm
        return arm

    def _get_arm(self, arm_id: str) -> dict:
        return self._arms.get(arm_id) or self._init_arm(arm_id)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, arm_id: str, context: np.ndarray) -> float:
        """
        Computes the UCB score for one arm given a context vector.

        Score = θ_a^T x + α √(x^T A_a^{-1} x)

        The second term is the exploration bonus. It is large when A_a
        has high uncertainty (few observations) and shrinks as the arm
        accumulates data.

        A_inv is computed fresh on every call. This is correct and simple.
        For production with many arms and high QPS, cache A_inv and
        invalidate on flush — documented in docs/adr/001-linucb-design.md.
        """
        arm = self._get_arm(arm_id)
        A_inv = np.linalg.inv(arm["A"])
        theta = A_inv @ arm["b"]

        exploitation = theta @ context
        exploration = self.alpha * np.sqrt(context @ A_inv @ context)

        return float(exploitation + exploration)

    def rank(self, candidate_arm_ids: list, context: np.ndarray) -> list:
        """
        Ranks all candidate arms by UCB score, descending.

        Args:
            candidate_arm_ids: product_ids to rank
            context: assembled context vector (output of context_builder)

        Returns:
            List of (arm_id, score) sorted descending. Caller takes first K.
        """
        if context.shape[0] != self.n_features:
            raise ValueError(
                f"Context vector has {context.shape[0]} features, "
                f"expected {self.n_features}."
            )

        scored = [
            (arm_id, self.score(arm_id, context))
            for arm_id in candidate_arm_ids
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ── Session buffer ────────────────────────────────────────────────────────

    def log(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """
        Buffers one interaction. Does not update the model immediately.
        Call flush() at session end to apply all buffered interactions.

        Args:
            arm_id:  product_id of the recommended arm
            context: context vector at recommendation time (copy stored)
            reward:  scalar reward observed (click=1, purchase=20, etc.)
        """
        self._buffer.append((arm_id, context.copy(), float(reward)))

    def flush(self) -> int:
        """
        Applies all buffered interactions to the model in one pass.

        Update rule per buffered interaction (arm_id, x, r):
            A_a ← A_a + x x^T
            b_a ← b_a + r · x

        Returns:
            Number of interactions applied.
        """
        n = len(self._buffer)
        if n == 0:
            return 0

        for arm_id, context, reward in self._buffer:
            arm = self._get_arm(arm_id)
            arm["A"] += np.outer(context, context)
            arm["b"] += reward * context

        self.total_interactions += n
        self._buffer.clear()
        return n

    # ── Model introspection ───────────────────────────────────────────────────

    def theta(self, arm_id: str) -> np.ndarray:
        """
        Returns the current weight vector θ_a = A_a^{-1} b_a for an arm.
        Used in notebooks for weight visualization.
        """
        arm = self._get_arm(arm_id)
        return np.linalg.inv(arm["A"]) @ arm["b"]

    def exploration_bonus(self, arm_id: str, context: np.ndarray) -> float:
        """
        Returns the exploration term α √(x^T A_a^{-1} x) in isolation.
        Useful for the Visibility Score feature in the seller analytics tab.
        A high bonus means LinUCB is still exploring this arm — low confidence.
        A low bonus means the model has converged on this arm.
        """
        arm = self._get_arm(arm_id)
        A_inv = np.linalg.inv(arm["A"])
        return float(self.alpha * np.sqrt(context @ A_inv @ context))

    def arm_count(self) -> int:
        """Returns the number of arms the model has seen so far."""
        return len(self._arms)

    def buffer_size(self) -> int:
        """Returns the number of interactions pending in the session buffer."""
        return len(self._buffer)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serializes full model state to disk.

        Saved: A matrices, b vectors, alpha, n_features, total_interactions.
        Not saved: session buffer (always flush before saving).
        """
        if self._buffer:
            raise RuntimeError(
                "Flush the session buffer before saving. "
                "Call flush() first, then save()."
            )
        state = {
            "n_features": self.n_features,
            "alpha": self.alpha,
            "arms": self._arms,
            "total_interactions": self.total_interactions,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "LinUCB":
        """
        Loads a saved model from disk. Returns a new LinUCB instance
        with restored arm state.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_features=state["n_features"], alpha=state["alpha"])
        obj._arms = state["arms"]
        obj.total_interactions = state["total_interactions"]
        return obj