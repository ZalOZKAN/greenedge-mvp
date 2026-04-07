"""Default configuration for the GreenEdge simulator & reward."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Coefficients for the multi-objective reward function.

    reward = -(alpha * energy_norm + beta * latency_norm + gamma * sla_penalty * sla_penalty_scale)
    """
    alpha: float = 0.35             # energy weight
    beta: float = 0.55              # latency weight
    gamma: float = 0.10             # SLA violation penalty weight
    sla_penalty_scale: float = 1.0  # multiplier on SLA penalty (1=binary, 3=×3, 5=×5…)


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    episode_length: int = 50
    sla_ms: float = 120.0          # SLA latency threshold (ms)
    seed: int = 42
    reward: RewardWeights = field(default_factory=RewardWeights)
