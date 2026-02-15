"""Default configuration for the GreenEdge simulator & reward."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Coefficients for the multi-objective reward function.

    reward = -(alpha * energy_norm + beta * latency_norm + gamma * sla_penalty)
    """
    alpha: float = 0.35   # energy weight
    beta: float = 0.55    # latency weight
    gamma: float = 0.10   # SLA violation penalty weight


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    episode_length: int = 50
    sla_ms: float = 120.0          # SLA latency threshold (ms)
    seed: int = 42
    reward: RewardWeights = field(default_factory=RewardWeights)
