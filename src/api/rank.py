"""
rank.py

POST /rank — returns a ranked list of product IDs for a given user context.

Cohort assignment:
  Logged-in users:  hash(user_id) % 5 == 0  → Greedy (control, ~20%)
                    hash(user_id) % 5 != 0   → LinUCB (treatment, ~80%)
  Guests:           session_id hash, same rule. Assignment is stable
                    within a session and random across sessions for guests.

The handler builds one context vector per candidate product, scores each
arm under the assigned policy, and returns the ranked product IDs.

The Next.js proxy at /api/recommendations calls this endpoint. The browser
never calls this URL directly.
"""

import hashlib
from datetime import datetime

from fastapi import APIRouter, Request

from features.context_builder import ContextBuilder
from .schemas import RankRequest, RankResponse

router = APIRouter()


def _assign_cohort(identifier: str) -> tuple[str, str]:
    """
    Deterministic cohort assignment from a string identifier.
    Returns (cohort_label, served_by) where:
      cohort_label: 'treatment' | 'control'
      served_by:    'linucb'   | 'greedy'
    """
    h = int(hashlib.sha256(identifier.encode()).hexdigest(), 16)
    if h % 5 == 0:
        return "control", "greedy"
    return "treatment", "linucb"


@router.post("/rank", response_model=RankResponse)
async def rank(request: Request, body: RankRequest) -> RankResponse:
    """
    Ranks candidate products for a user.

    Uses user_id for logged-in users and session_id for guests.
    Cohort assignment is stable per user_id and random-per-session for guests.
    """
    linucb = request.app.state.linucb
    greedy = request.app.state.greedy

    identifier = body.user_id or body.session_id
    cohort, served_by = _assign_cohort(identifier)

    # Build a context vector for each candidate product
    # time_of_day derived from server timestamp — request time, not client time
    ts = datetime.now()

    arm_ids     = []
    arm_scores  = []

    for product in body.candidate_products:
        ctx = ContextBuilder.build(
            timestamp=ts,
            device_type=body.context.device_type,
            category_affinity=body.context.category_affinity,
            session_depth=body.context.session_depth,
            price_tier=product.price_tier,
            product_category=product.category,
            seller_quality_score=product.seller_quality_score,
            days_since_listed=product.days_since_listed,
            seller_delivery_reliability=product.seller_delivery_reliability,
        )
        arm_ids.append(product.product_id)

        if served_by == "linucb":
            score = linucb.score(product.product_id, ctx)
        else:
            score = greedy._mean_reward(product.product_id)

        arm_scores.append((product.product_id, score))

    arm_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [pid for pid, _ in arm_scores]

    return RankResponse(
        ranked_product_ids=ranked_ids,
        served_by=served_by,
        cohort=cohort,
    )