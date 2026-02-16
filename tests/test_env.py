"""Tests for GreenEdge simulator environment."""

import numpy as np
import pytest

from greenedge.simulator.config import EnvConfig, RewardWeights
from greenedge.simulator.env import ACTION_LABELS, GreenEdgeEnv


class TestEnvConfig:
    """Test environment configuration."""

    def test_default_config(self):
        cfg = EnvConfig()
        assert cfg.episode_length == 50
        assert cfg.sla_ms == 120.0
        assert cfg.seed == 42

    def test_reward_weights_sum(self):
        rw = RewardWeights()
        total = rw.alpha + rw.beta + rw.gamma
        assert abs(total - 1.0) < 0.01, "Reward weights should sum to ~1.0"

    def test_custom_config(self):
        cfg = EnvConfig(episode_length=100, sla_ms=150.0, seed=123)
        assert cfg.episode_length == 100
        assert cfg.sla_ms == 150.0
        assert cfg.seed == 123


class TestGreenEdgeEnv:
    """Test GreenEdge environment."""

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(seed=42)
        return GreenEdgeEnv(config=cfg)

    def test_observation_space(self, env):
        assert env.observation_space.shape == (6,)
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 1.0

    def test_action_space(self, env):
        assert env.action_space.n == 3

    def test_action_labels(self):
        assert ACTION_LABELS[0] == "edge-a"
        assert ACTION_LABELS[1] == "edge-b"
        assert ACTION_LABELS[2] == "cloud"

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset()
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert "t" in info

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=100)
        obs2, _ = env.reset(seed=100)
        np.testing.assert_array_equal(obs1, obs2)

    def test_step_returns_correct_structure(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check info keys
        required_keys = ["t", "target", "latency_ms", "energy_per_mbps", "sla_violation"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_episode_terminates(self, env):
        env.reset()
        for _ in range(env.cfg.episode_length):
            _, _, terminated, _, _ = env.step(0)
        assert terminated, "Episode should terminate after episode_length steps"

    def test_all_actions_valid(self, env):
        env.reset()
        for action in range(3):
            env.reset()
            obs, reward, _, _, info = env.step(action)
            assert info["target"] == ACTION_LABELS[action]

    def test_sla_violation_binary(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["sla_violation"] in [0, 1]

    def test_latency_positive(self, env):
        env.reset()
        for action in range(3):
            env.reset()
            _, _, _, _, info = env.step(action)
            assert info["latency_ms"] > 0

    def test_energy_positive(self, env):
        env.reset()
        for action in range(3):
            env.reset()
            _, _, _, _, info = env.step(action)
            assert info["energy_per_mbps"] > 0
