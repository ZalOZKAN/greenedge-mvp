"""GreenEdge-5G  FastAPI service.

Usage:
    python -m greenedge.api.main          # starts on http://localhost:8000
    python -m greenedge.api.main --port 9000

Endpoints:
    GET  /health     → {"status": "ok"}
    POST /decision   → action + confidence + predicted KPIs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from greenedge.logging_config import get_logger
from greenedge.settings import settings
from greenedge.rl.baselines import greedy_min_latency
from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import ACTION_LABELS, GreenEdgeEnv

logger = get_logger("api")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ObservationIn(BaseModel):
    """Input: 6-dim observation vector."""
    obs: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="[cpu_a, cpu_b, q_a, q_b, link_q, energy_price]  all in 0-1",
        examples=[[0.30, 0.52, 0.19, 0.21, 0.75, 0.30]],
    )


class PredictedKPIs(BaseModel):
    latency_ms: float
    energy_per_mbps: float
    sla_violation: int


class DecisionOut(BaseModel):
    action: int = Field(..., ge=0, le=2)
    action_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    fallback_used: bool
    predicted_kpis: PredictedKPIs


# ---------------------------------------------------------------------------
# App & state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GreenEdge-5G Decision API",
    version="0.1.0",
    description="RL-based workload routing decisions for edge/cloud.",
)

# Add security middleware (rate limiting + optional API key auth)
from starlette.middleware.base import BaseHTTPMiddleware
from greenedge.api.security import create_security_middleware

app.add_middleware(BaseHTTPMiddleware, dispatch=create_security_middleware())


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


from starlette.responses import JSONResponse

# Global state – loaded once at startup
_model = None
_confidence_threshold = settings.api.confidence_threshold


def _get_model():
    """Lazy-load the SB3 model (PPO or DQN)."""
    global _model
    if _model is not None:
        return _model

    from stable_baselines3 import DQN, PPO

    repo_root = Path(__file__).resolve().parents[2]
    policy_path = repo_root / "experiments" / "policy"

    for cls in (PPO, DQN):
        try:
            _model = cls.load(str(policy_path))
            logger.info(f"Model loaded successfully: {cls.__name__}")
            return _model
        except FileNotFoundError:
            logger.debug(f"Policy file not found for {cls.__name__}")
            continue
        except Exception as e:
            logger.warning(f"Failed to load {cls.__name__}: {e}")
            continue
    logger.warning("No trained model available, using baseline policies")
    return None  # no model available → baselines only


def _predict_kpis(obs: np.ndarray, action: int) -> PredictedKPIs:
    """One-step simulation to estimate KPIs for the chosen action."""
    cfg = EnvConfig(seed=0)
    env = GreenEdgeEnv(config=cfg)
    # Manually set internal state from observation
    env._cpu_a = float(obs[0])
    env._cpu_b = float(obs[1])
    env._queue_a = float(obs[2])
    env._queue_b = float(obs[3])
    env._link_quality = float(obs[4])
    env._energy_price = float(obs[5])
    env._t = 0

    _, _, _, _, info = env.step(action)
    return PredictedKPIs(
        latency_ms=info["latency_ms"],
        energy_per_mbps=info["energy_per_mbps"],
        sla_violation=info["sla_violation"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    model = _get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/decision", response_model=DecisionOut)
def decision(body: ObservationIn):
    obs = np.array(body.obs, dtype=np.float32)

    model = _get_model()
    fallback_used = False

    if model is not None:
        import torch

        # Get action probabilities to derive confidence
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().numpy()

        action = int(np.argmax(probs))
        confidence = float(probs[action])

        # Safety: fall back to baseline if confidence is too low
        if confidence < _confidence_threshold:
            action = greedy_min_latency(obs)
            confidence = round(confidence, 4)
            fallback_used = True
    else:
        # No trained model available → pure baseline
        action = greedy_min_latency(obs)
        confidence = 0.0
        fallback_used = True

    predicted = _predict_kpis(obs, action)

    return DecisionOut(
        action=action,
        action_label=ACTION_LABELS[action],
        confidence=round(confidence, 4),
        fallback_used=fallback_used,
        predicted_kpis=predicted,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GreenEdge API server")
    parser.add_argument("--host", default=settings.api.host)
    parser.add_argument("--port", type=int, default=settings.api.port)
    args = parser.parse_args()

    import uvicorn
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "greenedge.api.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
