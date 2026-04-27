"""
greedy.py

Greedy baseline bandit — always ranks arms by highest historical mean reward.

This is the A/B control arm (20% of traffic). It represents the naive policy
of recommending whatever has sold best historically, ignoring user context.

Key differences from LinUCB:
  - No exploration. It exploits the current best arm every time.
  - Context-free. The context vector is accepted for interface compatibility
    but is not used in scoring.
  - Immediate updates. log() applies the update at call time. flush() is a
    no-op. This matches the "always show bestsellers" semantics — new data
    should affect the next recommendation immediately.

New arms default to a score of 0.0 until at least one reward is logged.
This means unseen products rank at the bottom, producing a rich-get-richer
dynamic that LinUCB's exploration bonus is designed to correct.
"""

import pickle
import numpy as np
from pathlib import Path

from .base import BaseBandit


class GreedyBaseline(BaseBandit):
    """
    Context-free bandit. Ranks arms by empirical mean reward.

    Arms with no observed rewards score 0.0 and rank below all arms
    with positive mean reward. This is intentional: the greedy policy
    has no mechanism to discover new products, making it the correct
    foil for measuring the value of LinUCB's exploration bonus.
    """

    def __init__(self):
        # Per-arm stats. Keyed by arm_id.
        # Each entry: {'total_reward': float, 'count': int}
        self._arms: dict = {}
        self.total_interactions: int = 0

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _mean_reward(self, arm_id: str) -> float:
        stats = self._arms.get(arm_id)
        if stats is None or stats["count"] == 0:
            return 0.0
        return stats["total_reward"] / stats["count"]

    def rank(self, candidate_arm_ids: list, context: np.ndarray = None) -> list:
        """
        Ranks candidate arms by empirical mean reward, descending.
        Context is accepted but ignored.
        """
        scored = [(arm_id, self._mean_reward(arm_id)) for arm_id in candidate_arm_ids]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ── Update ────────────────────────────────────────────────────────────────

    def log(self, arm_id: str, context: np.ndarray = None, reward: float = 0.0) -> None:
        """
        Applies the update immediately. No buffer.
        Context is accepted for interface compatibility but ignored.
        """
        if arm_id not in self._arms:
            self._arms[arm_id] = {"total_reward": 0.0, "count": 0}
        self._arms[arm_id]["total_reward"] += float(reward)
        self._arms[arm_id]["count"] += 1
        self.total_interactions += 1

    def flush(self) -> int:
        """No-op. Greedy updates are immediate. Returns 0."""
        return 0

    # ── Introspection ─────────────────────────────────────────────────────────

    def arm_count(self) -> int:
        return len(self._arms)

    def top_arms(self, k: int = 10) -> list:
        """Returns the top-k arms by mean reward. Useful for logging."""
        ranked = sorted(
            self._arms.items(),
            key=lambda x: x[1]["total_reward"] / max(x[1]["count"], 1),
            reverse=True,
        )
        return [(arm_id, stats) for arm_id, stats in ranked[:k]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        state = {
            "arms": self._arms,
            "total_interactions": self.total_interactions,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "GreedyBaseline":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._arms = state["arms"]
        obj.total_interactions = state["total_interactions"]
        return obj