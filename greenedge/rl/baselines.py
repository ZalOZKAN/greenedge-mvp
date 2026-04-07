"""Hand-crafted baseline policies for comparison with RL.

Each function takes an observation vector (6-dim) and returns an action (0/1/2).
Observation layout: [cpu_a, cpu_b, q_a, q_b, link_q, energy_price]

Baselines (weakest → strongest):
  random_policy      – uniform random action (lower bound reference)
  greedy_min_energy  – always pick minimum energy target
  simple_threshold   – CPU-based threshold heuristic
  greedy_min_latency – always pick minimum latency target
  weighted_heuristic – linear combination of latency + energy scores
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Core baselines
# ---------------------------------------------------------------------------

def greedy_min_latency(obs: np.ndarray) -> int:
    """Pick the target with expected lowest latency.

    Edge nodes are faster when their CPU load is low.
    Cloud has a high base latency (~65 ms) so prefer edges unless overloaded.
    """
    cpu_a, cpu_b, q_a, q_b, link_q, _ = obs.tolist()

    # Approximate latency scores (simplified model mirror)
    lat_a = 25.0 + 130.0 * (cpu_a ** 1.6) + 40.0 * q_a
    lat_b = 30.0 + 120.0 * (cpu_b ** 1.7) + 40.0 * q_b
    lat_cloud = 65.0 + 15.0 + max(0.0, 1.0 - link_q) * 30.0

    scores = [lat_a, lat_b, lat_cloud]
    return int(np.argmin(scores))


def greedy_min_energy(obs: np.ndarray) -> int:
    """Pick the target with expected lowest energy cost."""
    cpu_a, cpu_b, _, _, _, energy_price = obs.tolist()
    price = max(energy_price, 0.1)

    eng_a = (0.8 + 1.2 * cpu_a) * price
    eng_b = (0.75 + 1.3 * cpu_b) * price
    eng_cloud = (1.1 + 0.4 + 1.0 * 0.3) * price  # transport overhead

    scores = [eng_a, eng_b, eng_cloud]
    return int(np.argmin(scores))


def simple_threshold(obs: np.ndarray, cpu_thresh: float = 0.60) -> int:
    """If edge-a CPU is high, try edge-b; if both high, go cloud."""
    cpu_a, cpu_b, *_ = obs.tolist()

    if cpu_a < cpu_thresh:
        return 0   # edge-a
    elif cpu_b < cpu_thresh:
        return 1   # edge-b
    else:
        return 2   # cloud


# ---------------------------------------------------------------------------
# Additional baselines (lower-bound and weighted trade-off)
# ---------------------------------------------------------------------------

# Module-level RNG for random_policy — seeded per evaluation run via numpy global
_rng = np.random.default_rng()


def random_policy(obs: np.ndarray) -> int:  # noqa: ARG001
    """Uniform random action — the weakest possible baseline.

    Useful as a lower-bound reference: any reasonable policy should comfortably
    outperform random on both reward and SLA violation rate.
    The RNG is seeded externally (numpy global) to keep evaluation reproducible.
    """
    return int(_rng.integers(0, 3))


def weighted_heuristic(
    obs: np.ndarray,
    alpha: float = 0.35,
    beta: float = 0.55,
) -> int:
    """Combined latency + energy weighted heuristic (mirrors reward structure).

    Scores each target by a linear combination of approximate latency and energy
    using the same alpha/beta weights as the reward function (default: 0.35/0.55).
    This is the strongest purely hand-crafted baseline: it knows the same trade-off
    the RL reward optimises, but has no temporal memory or learning.

    Args:
        obs:   6-dim observation vector.
        alpha: Energy weight (mirrors RewardWeights.alpha, default 0.35).
        beta:  Latency weight (mirrors RewardWeights.beta, default 0.55).
    """
    cpu_a, cpu_b, q_a, q_b, link_q, energy_price = obs.tolist()
    price = max(energy_price, 0.1)

    # approximate latency (same formula as env)
    lq_penalty = max(0.0, 1.0 - link_q) * 30.0
    lat_a = (25.0 + 130.0 * (cpu_a ** 1.6) + 40.0 * q_a + lq_penalty) / 200.0
    lat_b = (30.0 + 120.0 * (cpu_b ** 1.7) + 40.0 * q_b + lq_penalty) / 200.0
    lat_cloud = (65.0 + 15.0 + lq_penalty + 6.0) / 200.0  # extra WAN jitter mean

    # approximate energy (same formula as env), normalised by /3.0
    eng_a = (0.8 + 1.2 * cpu_a) * price / 3.0
    eng_b = (0.75 + 1.3 * cpu_b) * price / 3.0
    eng_cloud = (1.1 + 0.4 + 0.3) * price / 3.0

    # combined score (lower = better, matches reward sign)
    score_a = alpha * eng_a + beta * lat_a
    score_b = alpha * eng_b + beta * lat_b
    score_cloud = alpha * eng_cloud + beta * lat_cloud

    return int(np.argmin([score_a, score_b, score_cloud]))
