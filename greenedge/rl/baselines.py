"""Hand-crafted baseline policies for comparison with RL.

Each function takes an observation vector (6-dim) and returns an action (0/1/2).
Observation layout: [cpu_a, cpu_b, q_a, q_b, link_q, energy_price]
"""

from __future__ import annotations

import numpy as np


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
