"""GreenEdge-5G  Gymnasium environment.

Observation (6-dim, float32, all roughly 0-1):
    [cpu_a, cpu_b, q_a, q_b, link_q, energy_price]

Actions (Discrete 3):
    0 = route to edge-a
    1 = route to edge-b
    2 = route to cloud

Reward:
    reward = -(alpha * energy_norm + beta * latency_norm + gamma * sla_penalty)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from greenedge.simulator.config import EnvConfig, RewardWeights

# Human-readable action labels
ACTION_LABELS = {0: "edge-a", 1: "edge-b", 2: "cloud"}


class GreenEdgeEnv(gym.Env):
    """Lightweight 5G edge/cloud workload-routing simulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rw: RewardWeights = self.cfg.reward

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Internal RNG
        self._rng = np.random.default_rng(self.cfg.seed)

        # State variables (set properly in reset)
        self._cpu_a: float = 0.0
        self._cpu_b: float = 0.0
        self._queue_a: float = 0.0
        self._queue_b: float = 0.0
        self._link_quality: float = 0.0
        self._energy_price: float = 0.0
        self._t: int = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self._cpu_a,
                self._cpu_b,
                self._queue_a,
                self._queue_b,
                self._link_quality,
                self._energy_price,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._cpu_a = self._clip01(0.30 + self._rng.normal(0, 0.08))
        self._cpu_b = self._clip01(0.50 + self._rng.normal(0, 0.08))
        self._queue_a = self._clip01(0.20 + self._rng.normal(0, 0.05))
        self._queue_b = self._clip01(0.25 + self._rng.normal(0, 0.05))
        self._link_quality = self._clip01(0.80 + self._rng.normal(0, 0.10))
        self._energy_price = self._clip01(0.40 + self._rng.normal(0, 0.10))
        self._t = 0

        return self._obs(), {"t": self._t}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._t += 1
        target = ACTION_LABELS[action]

        # --- dynamics: natural drift + workload pressure ---
        self._cpu_a = self._clip01(self._cpu_a + self._rng.normal(0, 0.03) - 0.01)
        self._cpu_b = self._clip01(self._cpu_b + self._rng.normal(0, 0.03) - 0.01)
        self._queue_a = self._clip01(self._queue_a + self._rng.normal(0, 0.02) - 0.005)
        self._queue_b = self._clip01(self._queue_b + self._rng.normal(0, 0.02) - 0.005)
        self._link_quality = self._clip01(self._link_quality + self._rng.normal(0, 0.04))
        self._energy_price = self._clip01(self._energy_price + self._rng.normal(0, 0.03))

        # apply workload to selected target
        if action == 0:  # edge-a
            self._cpu_a = self._clip01(self._cpu_a + 0.12)
            self._queue_a = self._clip01(self._queue_a + 0.08)
        elif action == 1:  # edge-b
            self._cpu_b = self._clip01(self._cpu_b + 0.12)
            self._queue_b = self._clip01(self._queue_b + 0.08)
        else:  # cloud
            # cloud absorbs easier but link quality matters
            pass

        # --- latency model (ms) ---
        latency_ms = self._compute_latency(action)

        # --- energy model (relative units per Mbps) ---
        energy = self._compute_energy(action)

        # --- SLA ---
        sla_violation = 1 if latency_ms > self.cfg.sla_ms else 0

        # --- reward ---
        latency_norm = latency_ms / 200.0
        energy_norm = energy / 3.0
        reward = -(
            self.rw.alpha * energy_norm
            + self.rw.beta * latency_norm
            + self.rw.gamma * sla_violation
        )

        terminated = self._t >= self.cfg.episode_length
        truncated = False

        info: Dict[str, Any] = {
            "t": self._t,
            "target": target,
            "latency_ms": round(float(latency_ms), 2),
            "energy_per_mbps": round(float(energy), 4),
            "sla_ms": self.cfg.sla_ms,
            "sla_violation": sla_violation,
            "loads": {
                "cpu_a": round(self._cpu_a, 3),
                "cpu_b": round(self._cpu_b, 3),
                "queue_a": round(self._queue_a, 3),
                "queue_b": round(self._queue_b, 3),
            },
            "link_quality": round(self._link_quality, 3),
            "energy_price": round(self._energy_price, 3),
            "reward": round(float(reward), 4),
        }

        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Latency & energy sub-models
    # ------------------------------------------------------------------
    def _compute_latency(self, action: int) -> float:
        jitter = abs(self._rng.normal(0, 4.0))
        lq_penalty = max(0.0, 1.0 - self._link_quality) * 30.0  # bad link → more ms

        if action == 0:  # edge-a
            base = 25.0
            load_effect = 130.0 * (self._cpu_a ** 1.6)
            queue_effect = 40.0 * self._queue_a
        elif action == 1:  # edge-b
            base = 30.0
            load_effect = 120.0 * (self._cpu_b ** 1.7)
            queue_effect = 40.0 * self._queue_b
        else:  # cloud
            base = 65.0
            load_effect = 50.0 * 0.3  # cloud is elastic
            queue_effect = 0.0
            jitter += abs(self._rng.normal(0, 6.0))  # extra WAN jitter

        return base + load_effect + queue_effect + lq_penalty + jitter

    def _compute_energy(self, action: int) -> float:
        price = max(self._energy_price, 0.1)
        if action == 0:
            return (0.8 + 1.2 * self._cpu_a) * price
        elif action == 1:
            return (0.75 + 1.3 * self._cpu_b) * price
        else:
            return (1.1 + 0.4 + 1.0 * 0.3) * price  # transport overhead
