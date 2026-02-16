"""Train an RL agent on the GreenEdge-5G environment.

Usage:
    python -m greenedge.rl.train --algo ppo --steps 20000
    python -m greenedge.rl.train --algo dqn --steps 50000

Saves the trained policy under /experiments/policy.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from greenedge.logging_config import get_logger
from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import GreenEdgeEnv

logger = get_logger("train")

# ---------------------------------------------------------------------------
# Gymnasium registration helper  (wraps our env so SB3 can use gym.make)
# ---------------------------------------------------------------------------

def _make_env(seed: int = 42) -> GreenEdgeEnv:
    cfg = EnvConfig(seed=seed)
    return GreenEdgeEnv(config=cfg)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(algo: str, total_timesteps: int, out_dir: Path, seed: int = 42) -> Path:
    """Train and save a policy. Returns path to saved model."""
    # Lazy import so module loads even without SB3 installed
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.env_util import make_vec_env

    algo = algo.lower()
    if algo not in ("ppo", "dqn"):
        raise ValueError(f"Unsupported algo '{algo}'. Choose ppo or dqn.")

    logger.info(f"Starting training: algo={algo}, steps={total_timesteps}, seed={seed}")

    # make_vec_env expects a callable
    vec_env = make_vec_env(lambda: _make_env(seed), n_envs=1, seed=seed)

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )
    else:  # dqn
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=500,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
        )

    model.learn(total_timesteps=total_timesteps)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "policy"
    model.save(str(save_path))
    print(f"[train] Policy saved → {save_path}.zip")

    vec_env.close()
    return Path(f"{save_path}.zip")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GreenEdge RL trainer")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"],
                    help="RL algorithm (default: ppo)")
    p.add_argument("--steps", type=int, default=20_000,
                    help="Total training timesteps (default: 20000)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    p.add_argument("--out", type=str, default=None,
                    help="Output directory (default: <repo>/experiments)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        # Default: <repo_root>/experiments
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = repo_root / "experiments"

    policy_path = train(
        algo=args.algo,
        total_timesteps=args.steps,
        out_dir=out_dir,
        seed=args.seed,
    )
    print(f"[done] {policy_path}")


if __name__ == "__main__":
    main()
