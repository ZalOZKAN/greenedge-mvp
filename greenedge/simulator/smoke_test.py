"""Smoke test for the GreenEdge simulator.

Run:  python -m greenedge.simulator.smoke_test

Expected output:
  - 1 sample observation (6-dim vector)
  - 3 sample decisions (one per action) with latency / energy / reward
"""

from __future__ import annotations

import textwrap

from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import ACTION_LABELS, GreenEdgeEnv


def _fmt_obs(obs) -> str:
    names = ["cpu_a", "cpu_b", "q_a", "q_b", "link_q", "energy_price"]
    parts = [f"  {n:>13s} = {v:.4f}" for n, v in zip(names, obs)]
    return "\n".join(parts)


def main() -> None:
    cfg = EnvConfig(seed=7)
    env = GreenEdgeEnv(config=cfg)
    obs, _ = env.reset()

    print("=" * 52)
    print("  GreenEdge-5G  Smoke Test")
    print("=" * 52)

    print("\n[1] Sample observation after reset:")
    print(_fmt_obs(obs))

    print("\n[2] Stepping with each action (edge-a / edge-b / cloud):\n")
    header = f"  {'Action':<10} {'Latency(ms)':>12} {'Energy/Mbps':>12} {'SLA viol':>9} {'Reward':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for action_id in range(3):
        # reset before each action so they see the same starting state
        env.reset(seed=7)
        _, reward, _, _, info = env.step(action_id)

        label = ACTION_LABELS[action_id]
        lat = info["latency_ms"]
        eng = info["energy_per_mbps"]
        sla = info["sla_violation"]
        print(f"  {label:<10} {lat:>12.2f} {eng:>12.4f} {sla:>9d} {reward:>8.4f}")

    print("\n" + "=" * 52)
    print("  Smoke test passed ✓")
    print("=" * 52)


if __name__ == "__main__":
    main()
