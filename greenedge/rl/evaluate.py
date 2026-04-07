"""Evaluate RL policy vs baselines and produce results + plots.

Usage:
    # Single seed (backward-compatible)
    python -m greenedge.rl.evaluate --episodes 200 --seed 0

    # Multi-seed robustness check (recommended for reports)
    python -m greenedge.rl.evaluate --episodes 200 --seeds 0 42 123

Outputs (under /experiments):
    results.json          – per-policy metrics for the primary seed
    results_summary.json  – per-seed + aggregate mean±std across all seeds
    plots_reward.png      – episode reward comparison (primary seed)
    plots_tradeoff.png    – latency vs energy scatter  (primary seed)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

# Patch random_policy RNG so it uses the seed passed to run_episodes
import greenedge.rl.baselines as _baselines_mod
from greenedge.logging_config import get_logger
from greenedge.rl.baselines import (
    greedy_min_energy,
    greedy_min_latency,
    random_policy,
    simple_threshold,
    weighted_heuristic,
)
from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import GreenEdgeEnv

logger = get_logger("evaluate")

# ---------------------------------------------------------------------------
# Runner: play N episodes with a given policy function
# ---------------------------------------------------------------------------

def run_episodes(
    policy_fn: Callable[[np.ndarray], int],
    n_episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Run *n_episodes* and collect KPI statistics."""
    # Seed the random_policy RNG for reproducibility
    _baselines_mod._rng = np.random.default_rng(seed)

    cfg = EnvConfig(seed=seed)
    env = GreenEdgeEnv(config=cfg)

    all_rewards: list[float] = []
    all_latencies: list[float] = []
    all_energies: list[float] = []
    all_sla: list[int] = []
    episode_rewards: list[float] = []

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

def plot_rewards(results: dict[str, dict], out_path: Path) -> None:
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


def plot_tradeoff(results: dict[str, dict], out_path: Path) -> None:
    """Latency vs Energy scatter with avg + p95 markers."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#795548"]
    markers = ["o", "s", "^", "D", "P", "X"]

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
# Multi-seed aggregate summary
# ---------------------------------------------------------------------------

AGGREGATE_KEYS = [
    "avg_reward", "std_reward", "avg_latency",
    "p95_latency", "avg_energy_per_mbps", "sla_violation_rate",
]


def _aggregate(per_seed: list[dict]) -> dict[str, Any]:
    """Compute mean and std across multiple seed runs for scalar KPIs."""
    agg: dict[str, Any] = {}
    for key in AGGREGATE_KEYS:
        vals = [s[key] for s in per_seed if key in s]
        if vals:
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    return agg


def evaluate_multiseed(
    n_episodes: int,
    policy_path: str | None,
    out_dir: Path,
    seeds: list[int],
) -> dict[str, Any]:
    """Run all policies for each seed; save per-seed + aggregate summary."""

    policies_base: dict[str, Callable[[np.ndarray], int]] = {
        "random_policy": random_policy,
        "greedy_min_energy": greedy_min_energy,
        "simple_threshold": simple_threshold,
        "greedy_min_latency": greedy_min_latency,
        "weighted_heuristic": weighted_heuristic,
    }

    if policy_path and Path(policy_path).exists():
        print(f"[eval] Loading RL policy from {policy_path}")
        policies_base["rl_ppo"] = _load_sb3_policy(policy_path)
    else:
        print("[eval] No trained policy found – evaluating baselines only.")

    # { policy_name: [seed0_stats, seed1_stats, ...] }
    per_policy_per_seed: dict[str, list[dict]] = {name: [] for name in policies_base}

    for seed in seeds:
        print(f"\n[eval] === Seed {seed} ===")
        for name, fn in policies_base.items():
            print(f"[eval] Running {name} for {n_episodes} episodes (seed={seed}) …")
            stats = run_episodes(fn, n_episodes, seed=seed)
            per_policy_per_seed[name].append({"seed": seed, **stats})

    # Build summary
    summary: dict[str, Any] = {
        "meta": {
            "n_episodes": n_episodes,
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "per_seed": {},
        "aggregate": {},
    }

    for name, seed_results in per_policy_per_seed.items():
        # strip episode_rewards from per_seed output to keep file small
        summary["per_seed"][name] = [
            {k: v for k, v in s.items() if k != "episode_rewards"}
            for s in seed_results
        ]
        summary["aggregate"][name] = _aggregate(
            [{k: v for k, v in s.items() if k != "episode_rewards"} for s in seed_results]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Multi-seed summary -> {summary_path}")

    # Print aggregate table
    _print_aggregate_table(summary["aggregate"])

    return summary


def _print_aggregate_table(aggregate: dict[str, Any]) -> None:
    """Pretty-print the mean±std aggregate table to stdout."""
    print("\n" + "=" * 88)
    print(f"  {'Policy':<22} {'AvgRew±std':>14} {'AvgLat±std':>14} {'P95Lat±std':>14} {'SLA%±std':>12}")
    print("  " + "-" * 82)
    for name, agg in aggregate.items():
        rew_m = agg.get("avg_reward_mean", float("nan"))
        rew_s = agg.get("avg_reward_std", 0.0)
        lat_m = agg.get("avg_latency_mean", float("nan"))
        lat_s = agg.get("avg_latency_std", 0.0)
        p95_m = agg.get("p95_latency_mean", float("nan"))
        p95_s = agg.get("p95_latency_std", 0.0)
        sla_m = agg.get("sla_violation_rate_mean", float("nan")) * 100
        sla_s = agg.get("sla_violation_rate_std", 0.0) * 100
        print(
            f"  {name:<22} {rew_m:>7.3f}±{rew_s:<5.2f} "
            f"{lat_m:>7.2f}±{lat_s:<5.2f} "
            f"{p95_m:>7.2f}±{p95_s:<5.2f} "
            f"{sla_m:>5.1f}%±{sla_s:<4.1f}%"
        )
    print("=" * 88)


# ---------------------------------------------------------------------------
# Main evaluate pipeline (single seed, backward-compatible)
# ---------------------------------------------------------------------------

def evaluate(
    n_episodes: int,
    policy_path: str | None,
    out_dir: Path,
    seed: int = 0,
) -> dict[str, dict]:
    """Run all policies and save results + plots (single seed)."""

    policies: dict[str, Callable[[np.ndarray], int]] = {
        "random_policy": random_policy,
        "greedy_min_energy": greedy_min_energy,
        "simple_threshold": simple_threshold,
        "greedy_min_latency": greedy_min_latency,
        "weighted_heuristic": weighted_heuristic,
    }

    # Load trained RL policy if available
    if policy_path and Path(policy_path).exists():
        print(f"[eval] Loading RL policy from {policy_path}")
        policies["rl_ppo"] = _load_sb3_policy(policy_path)
    else:
        print("[eval] No trained policy found – evaluating baselines only.")

    results: dict[str, dict] = {}
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
    p.add_argument("--seed", type=int, default=0,
                    help="Primary seed (used when --seeds is not specified)")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="List of seeds for multi-seed evaluation (e.g. --seeds 0 42 123). "
             "When specified, also writes results_summary.json.",
    )
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

    seeds = args.seeds if args.seeds else None

    if seeds and len(seeds) > 1:
        # Multi-seed mode: run all seeds, write results_summary.json
        evaluate_multiseed(
            n_episodes=args.episodes,
            policy_path=policy_path,
            out_dir=out_dir,
            seeds=seeds,
        )
        # Also run primary seed to update results.json and plots
        evaluate(
            n_episodes=args.episodes,
            policy_path=policy_path,
            out_dir=out_dir,
            seed=seeds[0],
        )
    else:
        primary_seed = seeds[0] if seeds else args.seed
        evaluate(
            n_episodes=args.episodes,
            policy_path=policy_path,
            out_dir=out_dir,
            seed=primary_seed,
        )


if __name__ == "__main__":
    main()
