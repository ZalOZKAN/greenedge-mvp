#!/usr/bin/env python3
"""GreenEdge-5G Systematic Experiment Runner.

Runs a series of PPO training + evaluation experiments, logging all results
to experiments/experiment_log.csv.

Usage (from repo root):
    python experiments/run_experiments.py

Each experiment:
  1. Trains a PPO policy with given config
  2. Evaluates it against all baselines (200 eps, seed=0, seed=1)
  3. Appends a row to experiment_log.csv
  4. Saves the best policy to experiments/policy.zip

Takes ~30-60 min total depending on hardware.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# Ensure greenedge package is importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from greenedge.rl.baselines import greedy_min_energy, greedy_min_latency, simple_threshold
from greenedge.simulator.config import EnvConfig, RewardWeights
from greenedge.simulator.env import GreenEdgeEnv
import numpy as np

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
LOG_CSV = EXPERIMENTS_DIR / "experiment_log.csv"

# ---------------------------------------------------------------------------
# Experiment Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    steps: int
    alpha: float = 0.35
    beta: float = 0.55
    gamma: float = 0.10
    sla_penalty_scale: float = 1.0
    lr: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    gamma_rl: float = 0.99   # RL discount factor (different from reward gamma)
    train_seed: int = 42
    eval_seeds: tuple = (0, 1, 2)
    eval_episodes: int = 200


# ---------------------------------------------------------------------------
# Evaluation helper (inline, no subprocess)
# ---------------------------------------------------------------------------

def run_episodes(policy_fn, n_episodes: int, seed: int, rw: RewardWeights) -> dict[str, Any]:
    cfg = EnvConfig(seed=seed, reward=rw)
    env = GreenEdgeEnv(config=cfg)

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
    }


def evaluate_policy(policy_fn, cfg: ExperimentConfig, rw: RewardWeights) -> dict[str, float]:
    """Evaluate across multiple seeds and average."""
    seed_results = []
    for seed in cfg.eval_seeds:
        r = run_episodes(policy_fn, cfg.eval_episodes, seed, rw)
        seed_results.append(r)

    keys = ["avg_reward", "std_reward", "avg_latency", "p95_latency",
            "avg_energy_per_mbps", "sla_violation_rate"]
    averaged = {}
    for k in keys:
        vals = [s[k] for s in seed_results]
        averaged[k] = round(float(np.mean(vals)), 4)
    return averaged


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_ppo(cfg: ExperimentConfig, out_path: Path, rw: RewardWeights) -> Any:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    def _make():
        return GreenEdgeEnv(config=EnvConfig(seed=cfg.train_seed, reward=rw))

    vec_env = make_vec_env(_make, n_envs=1, seed=cfg.train_seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=cfg.train_seed,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        gamma=cfg.gamma_rl,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
    )
    model.learn(total_timesteps=cfg.steps)
    vec_env.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path.with_suffix("")))
    return model


def make_sb3_policy(model):
    def _predict(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _predict


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "name", "steps", "alpha", "beta", "gamma", "sla_penalty_scale",
    "lr", "n_steps", "batch_size", "ent_coef", "clip_range",
    "avg_reward", "std_reward", "avg_latency", "p95_latency",
    "avg_energy_per_mbps", "sla_violation_rate",
    "vs_baseline_sla_delta",   # PPO SLA - best_baseline SLA
    "baseline_best_sla",
    "train_seed", "eval_seeds", "elapsed_s",
]


def append_csv(row: dict[str, Any]) -> None:
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS: list[ExperimentConfig] = [
    # ---- Step scaling (default weights) ----
    ExperimentConfig("step_50k",  steps=50_000),
    ExperimentConfig("step_100k", steps=100_000),
    ExperimentConfig("step_200k", steps=200_000),
    ExperimentConfig("step_500k", steps=500_000),

    # ---- Reward weight tuning @ 100k ----
    ExperimentConfig("wd_setA_100k", steps=100_000,
                     alpha=0.30, beta=0.40, gamma=0.30),
    ExperimentConfig("wd_setB_100k", steps=100_000,
                     alpha=0.25, beta=0.35, gamma=0.40),
    ExperimentConfig("wd_setC_100k", steps=100_000,
                     alpha=0.20, beta=0.30, gamma=0.50),
    ExperimentConfig("wd_setD_100k", steps=100_000,
                     alpha=0.30, beta=0.50, gamma=0.20),

    # ---- SLA penalty scaling @ 100k, best-direction weights ----
    ExperimentConfig("sla_scale3_setB_100k", steps=100_000,
                     alpha=0.25, beta=0.35, gamma=0.40, sla_penalty_scale=3.0),
    ExperimentConfig("sla_scale5_setB_100k", steps=100_000,
                     alpha=0.25, beta=0.35, gamma=0.40, sla_penalty_scale=5.0),
    ExperimentConfig("sla_scale3_setC_100k", steps=100_000,
                     alpha=0.20, beta=0.30, gamma=0.50, sla_penalty_scale=3.0),

    # ---- PPO hyperparam variants @ 100k (default weights + scale=1) ----
    ExperimentConfig("hp_highent_100k", steps=100_000,
                     ent_coef=0.05),
    ExperimentConfig("hp_lowlr_100k", steps=100_000,
                     lr=1e-4),
    ExperimentConfig("hp_widestep_100k", steps=100_000,
                     n_steps=512, batch_size=128),
]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"[EXP] {cfg.name} | steps={cfg.steps} | "
          f"α={cfg.alpha} β={cfg.beta} γ={cfg.gamma} "
          f"sla_scale={cfg.sla_penalty_scale}")
    print(f"{'='*70}")

    rw = RewardWeights(
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        sla_penalty_scale=cfg.sla_penalty_scale,
    )

    policy_path = EXPERIMENTS_DIR / f"_tmp_{cfg.name}_policy.zip"
    t0 = time.time()

    # --- Train ---
    print(f"  Training PPO ({cfg.steps:,} steps)…")
    model = train_ppo(cfg, policy_path, rw)
    elapsed_train = time.time() - t0
    print(f"  Training done in {elapsed_train:.0f}s")

    # --- Eval PPO ---
    print(f"  Evaluating PPO (seeds={cfg.eval_seeds}, {cfg.eval_episodes} eps each)…")
    ppo_fn = make_sb3_policy(model)
    ppo_stats = evaluate_policy(ppo_fn, cfg, rw)

    # --- Eval baselines (same rw so rewards are comparable) ---
    baseline_stats = {}
    for bl_name, bl_fn in [
        ("greedy_min_latency", greedy_min_latency),
        ("greedy_min_energy", greedy_min_energy),
        ("simple_threshold", simple_threshold),
    ]:
        baseline_stats[bl_name] = evaluate_policy(bl_fn, cfg, rw)

    best_baseline_sla = min(v["sla_violation_rate"] for v in baseline_stats.values())
    sla_delta = round(ppo_stats["sla_violation_rate"] - best_baseline_sla, 4)

    elapsed_total = round(time.time() - t0, 1)

    row = {
        "name": cfg.name,
        "steps": cfg.steps,
        "alpha": cfg.alpha, "beta": cfg.beta, "gamma": cfg.gamma,
        "sla_penalty_scale": cfg.sla_penalty_scale,
        "lr": cfg.lr, "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size, "ent_coef": cfg.ent_coef,
        "clip_range": cfg.clip_range,
        **ppo_stats,
        "vs_baseline_sla_delta": sla_delta,
        "baseline_best_sla": best_baseline_sla,
        "train_seed": cfg.train_seed,
        "eval_seeds": str(cfg.eval_seeds),
        "elapsed_s": elapsed_total,
    }

    append_csv(row)

    print(f"  PPO → reward={ppo_stats['avg_reward']:.3f}  "
          f"latency={ppo_stats['avg_latency']:.1f}ms  "
          f"p95={ppo_stats['p95_latency']:.1f}ms  "
          f"SLA={ppo_stats['sla_violation_rate']*100:.1f}%  "
          f"(Δ vs best baseline: {sla_delta*100:+.1f}pp)")

    return {"cfg": cfg, "ppo": ppo_stats, "baselines": baseline_stats,
            "policy_path": policy_path, "elapsed": elapsed_total}


def main() -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    print(f"GreenEdge-5G Experiment Runner")
    print(f"Results → {LOG_CSV}")
    print(f"Running {len(EXPERIMENTS)} experiments…")

    all_results = []
    for cfg in EXPERIMENTS:
        result = run_experiment(cfg)
        all_results.append(result)

    # --- Find best PPO config ---
    best = min(all_results, key=lambda r: r["ppo"]["sla_violation_rate"])
    best_cfg = best["cfg"]
    best_policy_src = best["policy_path"]

    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_cfg.name}")
    print(f"  SLA violation: {best['ppo']['sla_violation_rate']*100:.2f}%")
    print(f"  Avg reward:    {best['ppo']['avg_reward']:.3f}")
    print(f"  Avg latency:   {best['ppo']['avg_latency']:.1f} ms")
    print(f"  P95 latency:   {best['ppo']['p95_latency']:.1f} ms")
    print(f"{'='*70}")

    # --- Promote best policy to experiments/policy.zip ---
    import shutil
    best_zip = best_policy_src.with_suffix(".zip") if not str(best_policy_src).endswith(".zip") else best_policy_src
    if best_zip.exists():
        dest = EXPERIMENTS_DIR / "policy.zip"
        shutil.copy2(best_zip, dest)
        print(f"[runner] Best policy saved → {dest}")

    # --- Re-run final evaluation with standard seed=0 and produce results.json ---
    print("\n[runner] Producing final results.json with best policy…")
    rw_best = RewardWeights(
        alpha=best_cfg.alpha, beta=best_cfg.beta,
        gamma=best_cfg.gamma, sla_penalty_scale=best_cfg.sla_penalty_scale,
    )

    # For final results.json, use DEFAULT reward weights so dashboard comparison is fair
    rw_default = RewardWeights()
    from stable_baselines3 import PPO as _PPO
    final_model = _PPO.load(str(EXPERIMENTS_DIR / "policy"))

    final_results = {}
    for name, fn in [
        ("greedy_min_latency", greedy_min_latency),
        ("greedy_min_energy", greedy_min_energy),
        ("simple_threshold", simple_threshold),
    ]:
        final_results[name] = run_episodes(fn, 200, 0, rw_default)
        del final_results[name]["avg_reward"]  # recomputed below with default weights
        # Re-add reward computed under default weights
        final_results[name] = run_episodes(fn, 200, 0, rw_default)

    ppo_fn = make_sb3_policy(final_model)
    final_results["rl_ppo"] = run_episodes(ppo_fn, 200, 0, rw_default)

    # Strip episode_rewards if present
    summary = {k: {m: v for m, v in s.items() if m != "episode_rewards"}
               for k, s in final_results.items()}

    results_path = EXPERIMENTS_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[runner] {results_path} updated.")

    # Print final table
    print("\n" + "=" * 72)
    print(f"  {'Policy':<22} {'AvgRew':>8} {'AvgLat':>8} {'P95Lat':>8} {'E/Mbps':>8} {'SLA%':>7}")
    print("  " + "-" * 66)
    order = ["rl_ppo", "greedy_min_latency", "simple_threshold", "greedy_min_energy"]
    for name in order:
        if name in summary:
            s = summary[name]
            print(f"  {name:<22} {s['avg_reward']:>8.3f} "
                  f"{s['avg_latency']:>8.2f} {s['p95_latency']:>8.2f} "
                  f"{s['avg_energy_per_mbps']:>8.4f} {s['sla_violation_rate']*100:>6.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
