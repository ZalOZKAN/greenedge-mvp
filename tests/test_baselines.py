"""Tests for baseline policies."""

import numpy as np
import pytest

from greenedge.rl.baselines import (
    greedy_min_energy,
    greedy_min_latency,
    simple_threshold,
)


class TestGreedyMinLatency:
    """Test greedy minimum latency policy."""

    def test_returns_valid_action(self):
        obs = np.array([0.3, 0.5, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        action = greedy_min_latency(obs)
        assert action in [0, 1, 2]

    def test_prefers_low_cpu_edge(self):
        # Edge-a has very low CPU, should be preferred
        obs = np.array([0.1, 0.9, 0.1, 0.1, 0.9, 0.3], dtype=np.float32)
        action = greedy_min_latency(obs)
        assert action == 0, "Should prefer edge-a when it has low CPU"

    def test_prefers_cloud_when_edges_overloaded(self):
        # Both edges overloaded
        obs = np.array([0.95, 0.95, 0.8, 0.8, 0.9, 0.3], dtype=np.float32)
        action = greedy_min_latency(obs)
        assert action == 2, "Should prefer cloud when edges are overloaded"


class TestGreedyMinEnergy:
    """Test greedy minimum energy policy."""

    def test_returns_valid_action(self):
        obs = np.array([0.3, 0.5, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        action = greedy_min_energy(obs)
        assert action in [0, 1, 2]

    def test_prefers_low_cpu_for_energy(self):
        # Edge-b has lower CPU, lower energy consumption
        obs = np.array([0.8, 0.2, 0.1, 0.1, 0.9, 0.5], dtype=np.float32)
        action = greedy_min_energy(obs)
        assert action == 1, "Should prefer edge-b when it has lower CPU"


class TestSimpleThreshold:
    """Test simple threshold policy."""

    def test_returns_valid_action(self):
        obs = np.array([0.3, 0.5, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        action = simple_threshold(obs)
        assert action in [0, 1, 2]

    def test_prefers_edge_a_when_low_cpu(self):
        obs = np.array([0.3, 0.7, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        action = simple_threshold(obs, cpu_thresh=0.6)
        assert action == 0, "Should prefer edge-a when its CPU is below threshold"

    def test_prefers_edge_b_when_a_high(self):
        obs = np.array([0.7, 0.4, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        action = simple_threshold(obs, cpu_thresh=0.6)
        assert action == 1, "Should prefer edge-b when edge-a CPU is high"

    def test_prefers_cloud_when_both_high(self):
        obs = np.array([0.8, 0.8, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        action = simple_threshold(obs, cpu_thresh=0.6)
        assert action == 2, "Should prefer cloud when both edges have high CPU"


class TestPolicyConsistency:
    """Test that policies are deterministic."""

    def test_greedy_latency_deterministic(self):
        obs = np.array([0.3, 0.5, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        actions = [greedy_min_latency(obs) for _ in range(10)]
        assert len(set(actions)) == 1, "Policy should be deterministic"

    def test_greedy_energy_deterministic(self):
        obs = np.array([0.3, 0.5, 0.2, 0.25, 0.8, 0.4], dtype=np.float32)
        actions = [greedy_min_energy(obs) for _ in range(10)]
        assert len(set(actions)) == 1, "Policy should be deterministic"
