"""
base.py

Abstract base class all bandit implementations must satisfy.

Every bandit exposes the same four-method contract:
  rank(candidate_arm_ids, context)  → ordered list of (arm_id, score)
  log(arm_id, context, reward)      → buffers one interaction
  flush()                           → applies buffer, returns count flushed
  save(path) / load(path)           → disk persistence

The separation between log() and flush() is intentional.
LinUCB uses session-level updates: interactions are buffered during a session
and applied in one pass at session end. Greedy updates immediately on log()
and treats flush() as a no-op. Both satisfy the same interface so the
evaluation harness can treat them identically.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseBandit(ABC):

    @abstractmethod
    def rank(self, candidate_arm_ids: list, context: "np.ndarray") -> list:
        """
        Scores and ranks candidate arms for a given context.

        Args:
            candidate_arm_ids: list of arm identifiers (e.g. product_ids)
            context: 1-D numpy array — the assembled context vector

        Returns:
            List of (arm_id, score) tuples, sorted descending by score.
            Callers take the first K items for display.
        """

    @abstractmethod
    def log(self, arm_id: str, context: "np.ndarray", reward: float) -> None:
        """
        Records one interaction. May buffer or apply immediately
        depending on the implementation.

        Args:
            arm_id:  identifier of the arm that was shown
            context: context vector at the time of the recommendation
            reward:  observed scalar reward
        """

    @abstractmethod
    def flush(self) -> int:
        """
        Applies any buffered interactions to the model.
        Call once at the end of each user session.

        Returns:
            Number of buffered interactions applied.
            Implementations that update immediately return 0.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializes model state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseBandit":
        """Deserializes model state from disk. Returns a new instance."""

    def snapshot(self, directory: str, name: str) -> Path:
        """
        Convenience wrapper. Saves to directory/name.pkl and returns the path.
        Creates the directory if it does not exist.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.pkl"
        self.save(str(path))
        return path