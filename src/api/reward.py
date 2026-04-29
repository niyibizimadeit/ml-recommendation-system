"""
reward.py

POST /reward — logs one interaction event and buffers it for the model.
POST /flush  — applies all buffered interactions for a session.

Update flow:
  1. Client calls POST /reward after each click, add_to_cart, or purchase.
     The interaction is buffered via linucb.log() — model NOT updated yet.
  2. Client calls POST /flush at session end (browser close, 30-min timeout,
     or explicit signal from Next.js middleware).
     linucb.flush() applies all buffered interactions in one locked write.

Greedy updates immediately on /reward (no buffer needed).
LinUCB buffers and applies on /flush.

The asyncio lock on flush() prevents concurrent session flushes from
corrupting the A matrices via simultaneous outer product writes.
"""

from datetime import datetime

from fastapi import APIRouter, Request

from features.context_builder import ContextBuilder
from .schemas import RewardRequest, RewardResponse, FlushRequest, FlushResponse

router = APIRouter()


@router.post("/reward", response_model=RewardResponse)
async def reward(request: Request, body: RewardRequest) -> RewardResponse:
    """
    Buffers one interaction for the LinUCB model.
    Updates Greedy immediately.

    Only call this for: click, add_to_cart, purchase, delivery_success, delivery_failure.
    Do not call for product_view — zero reward events don't move the model.
    """
    linucb = request.app.state.linucb
    greedy = request.app.state.greedy

    ts = datetime.now()

    ctx = ContextBuilder.build(
        timestamp=ts,
        device_type=body.context.device_type,
        category_affinity=body.context.category_affinity,
        session_depth=body.context.session_depth,
        price_tier=body.product.price_tier,
        product_category=body.product.category,
        seller_quality_score=body.product.seller_quality_score,
        days_since_listed=body.product.days_since_listed,
        seller_delivery_reliability=body.product.seller_delivery_reliability,
    )

    if body.served_by == "linucb":
        # Buffer — applied on /flush at session end
        linucb.log(body.product_id, ctx, body.reward)
        model_updated = False
    else:
        # Greedy updates immediately
        greedy.log(body.product_id, reward=body.reward)
        model_updated = True

    return RewardResponse(status="ok", model_updated=model_updated)


@router.post("/flush", response_model=FlushResponse)
async def flush(request: Request, body: FlushRequest) -> FlushResponse:
    """
    Applies all buffered LinUCB interactions for a session.

    Call this at session end. In production this is triggered by:
      - Explicit call from Next.js session-end middleware
      - Background task after 30-minute inactivity timeout

    The asyncio lock prevents concurrent flushes from corrupting A matrices.
    """
    linucb = request.app.state.linucb
    lock   = request.app.state.model_lock

    async with lock:
        n = linucb.flush()

    return FlushResponse(status="ok", interactions_applied=n)