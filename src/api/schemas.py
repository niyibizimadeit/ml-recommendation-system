"""
schemas.py

Pydantic models for all API request and response payloads.

Every field has an explicit type and a description. This drives
the auto-generated docs at /docs and enforces contract discipline
between the Next.js proxy and the FastAPI service.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class UserContext(BaseModel):
    time_of_day: float = Field(..., ge=0.0, le=1.0, description="Normalized hour [0, 1]")
    device_type: str   = Field(..., pattern="^(mobile|desktop)$")
    category_affinity: dict[str, float] = Field(
        default_factory=dict,
        description="Category name → affinity weight. Pass empty dict for new users."
    )
    session_depth: int = Field(..., ge=0, le=999)


class CandidateProduct(BaseModel):
    product_id: str
    price_tier: float = Field(..., ge=0.0, le=1.0)
    category: str
    seller_quality_score: float = Field(..., ge=0.0, le=1.0)
    days_since_listed: float = Field(..., ge=0.0, le=1.0)
    seller_delivery_reliability: float = Field(..., ge=0.0, le=1.0)


# ── POST /rank ────────────────────────────────────────────────────────────────

class RankRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="UUID for logged-in users. Null for guests.")
    session_id: str
    context: UserContext
    candidate_products: list[CandidateProduct] = Field(..., min_length=1)


class RankResponse(BaseModel):
    ranked_product_ids: list[str]
    served_by: str = Field(..., pattern="^(linucb|greedy)$")
    cohort: str    = Field(..., pattern="^(treatment|control)$")


# ── POST /reward ──────────────────────────────────────────────────────────────

class RewardRequest(BaseModel):
    session_id: str
    product_id: str
    event: str = Field(..., pattern="^(click|add_to_cart|purchase|delivery_success|delivery_failure)$")
    reward: float
    served_by: str = Field(..., pattern="^(linucb|greedy)$")
    context: UserContext
    product: CandidateProduct


class RewardResponse(BaseModel):
    status: str
    model_updated: bool


# ── POST /flush ───────────────────────────────────────────────────────────────

class FlushRequest(BaseModel):
    session_id: str


class FlushResponse(BaseModel):
    status: str
    interactions_applied: int


# ── GET /health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    linucb_arms: int
    linucb_interactions: int
    greedy_arms: int
    alpha: float