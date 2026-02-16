"""Evaluate RL policy vs baselines and produce results + plots.

Usage:
    python -m greenedge.rl.evaluate --episodes 200

Outputs (under /experiments):
    results.json        – per-policy metrics
    plots_reward.png    – episode reward comparison
    plots_tradeoff.png  – latency vs energy scatter (Pareto-style)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from greenedge.logging_config import get_logger
from greenedge.rl.baselines import greedy_min_energy, greedy_min_latency, simple_threshold
from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import ACTION_LABELS, GreenEdgeEnv

logger = get_logger("evaluate")

# ---------------------------------------------------------------------------
# Runner: play N episodes with a given policy function
# ---------------------------------------------------------------------------

def run_episodes(
    policy_fn: Callable[[np.ndarray], int],
    n_episodes: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run *n_episodes* and collect KPI statistics."""
    cfg = EnvConfig(seed=seed)
    env = GreenEdgeEnv(config=cfg)

    all_rewards: List[float] = []
    all_latencies: List[float] = []
    all_energies: List[float] = []
    all_sla: List[int] = []
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            all_latencies.append(info["latency_ms"])
            all_energies.append(info["energy_per_mbps"])
            all_sla.append(info["sla_violation"])
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        all_rewards.append(ep_reward)

    lat = np.array(all_latencies)
    eng = np.array(all_energies)
    sla = np.array(all_sla)

    return {
        "avg_reward": round(float(np.mean(episode_rewards)), 4),
        "std_reward": round(float(np.std(episode_rewards)), 4),
        "avg_latency": round(float(np.mean(lat)), 2),
        "p95_latency": round(float(np.percentile(lat, 95)), 2),
        "avg_energy_per_mbps": round(float(np.mean(eng)), 4),
        "sla_violation_rate": round(float(np.mean(sla)), 4),
        "episode_rewards": [round(r, 4) for r in episode_rewards],
    }


# ---------------------------------------------------------------------------
# SB3 policy wrapper
# ---------------------------------------------------------------------------

def _load_sb3_policy(policy_path: str):
    """Return a callable (obs -> action) from a saved SB3 model."""
    from stable_baselines3 import DQN, PPO

    # Try PPO first, then DQN
    for cls in (PPO, DQN):
        try:
            model = cls.load(policy_path)
            logger.info(f"Loaded {cls.__name__} model from {policy_path}")
            def _predict(obs: np.ndarray, _m=model) -> int:
                action, _ = _m.predict(obs, deterministic=True)
                return int(action)
            return _predict
        except FileNotFoundError:
            logger.debug(f"Policy file not found for {cls.__name__}")
            continue
        except Exception as e:
            logger.warning(f"Failed to load {cls.__name__}: {e}")
            continue
    raise RuntimeError(f"Could not load policy from {policy_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rewards(results: Dict[str, Dict], out_path: Path) -> None:
    """Bar + line chart comparing episode rewards across policies."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in results.items():
        rewards = data["episode_rewards"]
        ax.plot(rewards, label=name, alpha=0.8, linewidth=1.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Episode Reward – RL vs Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_tradeoff(results: Dict[str, Dict], out_path: Path) -> None:
    """Latency vs Energy scatter with avg + p95 markers."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    markers = ["o", "s", "^", "D"]

    for i, (name, data) in enumerate(results.items()):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.scatter(
            data["avg_energy_per_mbps"],
            data["avg_latency"],
            color=c, marker=m, s=120, zorder=5, label=f"{name} (avg)",
        )
        ax.scatter(
            data["avg_energy_per_mbps"],
            data["p95_latency"],
            color=c, marker=m, s=60, zorder=5, alpha=0.5,
            label=f"{name} (p95)",
        )
        # connect avg to p95
        ax.plot(
            [data["avg_energy_per_mbps"], data["avg_energy_per_mbps"]],
            [data["avg_latency"], data["p95_latency"]],
            color=c, linestyle="--", linewidth=1, alpha=0.6,
        )

    ax.set_xlabel("Energy / Mbps (relative)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Energy Trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


# ---------------------------------------------------------------------------
# Main evaluate pipeline
# ---------------------------------------------------------------------------

def evaluate(
    n_episodes: int,
    policy_path: str | None,
    out_dir: Path,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run all policies and save results + plots."""

    policies: Dict[str, Callable[[np.ndarray], int]] = {
        "greedy_min_latency": greedy_min_latency,
        "greedy_min_energy": greedy_min_energy,
        "simple_threshold": simple_threshold,
    }

    # Load trained RL policy if available
    if policy_path and Path(policy_path).exists():
        print(f"[eval] Loading RL policy from {policy_path}")
        policies["rl_ppo"] = _load_sb3_policy(policy_path)
    else:
        print("[eval] No trained policy found – evaluating baselines only.")

    results: Dict[str, Dict] = {}
    for name, fn in policies.items():
        print(f"[eval] Running {name} for {n_episodes} episodes …")
        stats = run_episodes(fn, n_episodes, seed=seed)
        # Remove episode_rewards from summary (keep for plots)
        results[name] = stats

    # --- Save results.json (without episode_rewards list for cleanliness) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, data in results.items():
        summary[name] = {k: v for k, v in data.items() if k != "episode_rewards"}

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[eval] {results_path}")

    # --- Plots ---
    plot_rewards(results, out_dir / "plots_reward.png")
    plot_tradeoff(results, out_dir / "plots_tradeoff.png")

    # --- Print summary table ---
    print("\n" + "=" * 72)
    print(f"  {'Policy':<22} {'AvgRew':>8} {'AvgLat':>8} {'P95Lat':>8} {'E/Mbps':>8} {'SLA%':>7}")
    print("  " + "-" * 66)
    for name, s in summary.items():
        print(
            f"  {name:<22} {s['avg_reward']:>8.3f} "
            f"{s['avg_latency']:>8.2f} {s['p95_latency']:>8.2f} "
            f"{s['avg_energy_per_mbps']:>8.4f} {s['sla_violation_rate']*100:>6.1f}%"
        )
    print("=" * 72)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GreenEdge RL evaluator")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--policy", type=str, default=None,
                    help="Path to SB3 policy zip (default: experiments/policy.zip)")
    p.add_argument("--out", type=str, default=None,
                    help="Output dir (default: <repo>/experiments)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out) if args.out else repo_root / "experiments"

    policy_path = args.policy
    if policy_path is None:
        default = out_dir / "policy.zip"
        if default.exists():
            policy_path = str(default)

    evaluate(
        n_episodes=args.episodes,
        policy_path=policy_path,
        out_dir=out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
