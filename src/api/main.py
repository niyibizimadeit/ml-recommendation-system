"""
main.py

FastAPI application entry point.

Model lifecycle:
  - LinUCB and Greedy are module-level singletons loaded at startup.
  - Config is read from configs/config.yaml at startup.
  - A snapshot is saved to data/snapshots/linucb.pkl on shutdown.
  - On startup, if a snapshot exists it is loaded — model state survives restarts.

Endpoints:
  POST /rank    — rank candidate products for a user
  POST /reward  — buffer one interaction
  POST /flush   — apply buffered interactions at session end
  GET  /health  — model state + uptime check

Run locally:
  uvicorn src.api.main:app --reload --port 8000

Docs:
  http://localhost:8000/docs
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Allow running from repo root: uvicorn src.api.main:app
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bandits.linucb import LinUCB
from bandits.greedy import GreedyBaseline
from features.context_builder import N_FEATURES
from .rank import router as rank_router
from .reward import router as reward_router
from .schemas import HealthResponse

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH   = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
SNAPSHOT_PATH = Path(__file__).parent.parent.parent / "data" / "snapshots" / "linucb.pkl"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    # Fallback defaults so the server starts without a config file
    return {
        "bandit": {"alpha": 1.0},
        "rewards": {
            "click": 1, "add_to_cart": 5, "purchase": 20,
            "delivery_success": 3, "delivery_failure": -10,
        },
    }


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load config, restore snapshot if available, initialize models.
    Shutdown: flush buffer, save snapshot.
    """
    config = _load_config()
    alpha  = config.get("bandit", {}).get("alpha", 1.0)

    # Restore LinUCB from snapshot if one exists
    if SNAPSHOT_PATH.exists():
        linucb = LinUCB.load(str(SNAPSHOT_PATH))
        print(f"[startup] Restored LinUCB snapshot: {linucb.arm_count()} arms, "
              f"{linucb.total_interactions} interactions")
    else:
        linucb = LinUCB(n_features=N_FEATURES, alpha=alpha)
        print(f"[startup] Fresh LinUCB model — n_features={N_FEATURES}, alpha={alpha}")

    greedy = GreedyBaseline()
    lock   = asyncio.Lock()

    app.state.linucb      = linucb
    app.state.greedy      = greedy
    app.state.model_lock  = lock
    app.state.config      = config

    yield  # server is running

    # Shutdown: flush any pending buffer then save snapshot
    async with lock:
        n = linucb.flush()
        if n:
            print(f"[shutdown] Flushed {n} pending interactions before saving.")

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    linucb.save(str(SNAPSHOT_PATH))
    print(f"[shutdown] Snapshot saved to {SNAPSHOT_PATH}")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GiraXpress ML Service",
    description="LinUCB contextual bandit recommendation engine for GiraXpress.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(rank_router)
app.include_router(reward_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    linucb = app.state.linucb
    greedy = app.state.greedy
    config = app.state.config
    return HealthResponse(
        status="ok",
        linucb_arms=linucb.arm_count(),
        linucb_interactions=linucb.total_interactions,
        greedy_arms=greedy.arm_count(),
        alpha=config.get("bandit", {}).get("alpha", 1.0),
    )