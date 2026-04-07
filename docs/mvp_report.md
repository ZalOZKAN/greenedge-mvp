# GreenEdge-5G — MVP Technical Report

> **Scope:** This report documents simulation-based validation of the GreenEdge-5G RL decision engine.
> All metrics are produced in a Gymnasium simulation environment; no real 5G field data was used.

---

## 1. Overview

GreenEdge-5G is an RL-based decision engine that routes workloads across a heterogeneous 5G infrastructure consisting of two edge nodes (**edge-a**, **edge-b**) and a **cloud** backend. The system optimises a multi-objective function balancing **latency**, **energy consumption**, and **SLA compliance** using Proximal Policy Optimization (PPO).

### Architecture

```
                  ┌──────────┐
  Observation ──▶ │ RL Agent │──▶ Action (0/1/2)
  [6-dim vector]  │  (PPO)   │
                  └──────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌─────────┐  ┌──────────┐
    │ edge-a  │  │ edge-b  │  │  cloud   │
    └─────────┘  └─────────┘  └──────────┘
```

**Observation vector (6-dim):** `[cpu_a, cpu_b, queue_a, queue_b, link_quality, energy_price]`  
All dimensions normalised to [0, 1].

---

## 2. Reward Function

$$
r = -\bigl(\alpha \cdot E_{\text{norm}} + \beta \cdot L_{\text{norm}} + \gamma \cdot \text{SLA\_scale} \cdot \mathbb{1}_{\text{SLA}}\bigr)
$$

| Term | Formula | Purpose |
|------|---------|---------|
| $E_{\text{norm}}$ | `energy / 3.0` | Normalised energy consumption |
| $L_{\text{norm}}$ | `latency_ms / 200.0` | Normalised end-to-end latency |
| $\mathbb{1}_{\text{SLA}}$ | `1 if latency_ms > 120 else 0` | Hard SLA violation indicator |

**Default weights:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| α (alpha) | **0.35** | Energy is important but secondary |
| β (beta) | **0.55** | Latency is the primary SLA driver in 5G edge |
| γ (gamma) | **0.10** | SLA penalty weight |
| SLA_scale | **1.0** | Binary penalty; tuned to 3× or 5× in experiments |

**Design rationale:**
- β > α reflects the operational priority: in Private 5G for industrial automation, latency SLA compliance is more critical than marginal energy savings.
- The SLA indicator term is discrete (binary) to create a clear incentive boundary at 120 ms rather than encouraging gradual degradation.
- Weights were finalised via grid search over α∈{0.3, 0.35, 0.4}, β∈{0.5, 0.55, 0.6}, and SLA_scale∈{1, 3, 5}. The chosen configuration minimised SLA violations while keeping energy competitive with baselines.

---

## 3. Baselines

Six policies are evaluated for comparison, ordered from weakest to strongest:

| Policy | Type | Description |
|--------|------|-------------|
| `random_policy` | Stochastic lower bound | Uniform random action — establishes the floor |
| `greedy_min_energy` | Greedy heuristic | Always routes to minimum energy target |
| `simple_threshold` | Rule-based | Routes based on CPU load threshold (0.60) |
| `greedy_min_latency` | Greedy heuristic | Always routes to minimum latency target |
| `weighted_heuristic` | Informed heuristic | Linear combination of latency + energy (mirrors α/β weights) |
| `rl_ppo` | Learned policy | PPO trained 500k steps in simulation |

The `weighted_heuristic` is the "oracle heuristic" — it applies the same latency/energy trade-off as the reward function but without temporal state memory. It is the strongest hand-crafted baseline and the most meaningful comparison for demonstrating the value of RL.

---

## 4. KPI Results

### 4.1 Primary Evaluation (seed=0, 200 episodes)

Used for reproducible demos and dashboard display.

| Policy | Avg Reward | Avg Latency (ms) | P95 Latency (ms) | Energy/Mbps | SLA Violation% |
|--------|-----------|-------------------|-------------------|-------------|----------------|
| **rl_ppo** | **-17.09** | **93.60** | **107.61** | **0.7225** | **0.12%** |
| greedy_min_latency | -18.06 | 98.35 | 121.59 | 0.7280 | 5.79% |
| simple_threshold | -18.60 | 100.63 | 130.44 | 0.7085 | 12.56% |
| greedy_min_energy | -20.61 | 111.37 | 178.02 | 0.7023 | 24.03% |

> *Reproducible via `python -m greenedge.rl.evaluate --episodes 200 --seed 0`*
>
> *Full 6-policy table (including `random_policy` and `weighted_heuristic`) is available in `experiments/results_summary.json` after running `make multiseed`.*
>
> **Note:** `random_policy` and `weighted_heuristic` are included in evaluation scripts for benchmarking purposes but are not exposed in the dashboard UI for simplicity.

### 4.2 Multi-seed Evaluation (seeds {0, 42, 123}) — mean ± std

Used to demonstrate result stability across different environment initialisations.

| Policy | Avg Reward | Avg Latency | P95 Latency | SLA Violation% |
|--------|-----------|-------------|-------------|----------------|
| **rl_ppo** | **-17.01 ± 0.08** | **93.51 ± 0.10 ms** | **107.18 ± 0.42 ms** | **0.10% ± 0.02%** |
| weighted_heuristic | -18.05 ± 0.12 | 98.97 ± 0.06 ms | 122.63 ± 0.19 ms | 6.89% ± 0.20% |
| greedy_min_latency | -18.00 ± 0.12 | 98.32 ± 0.02 ms | 121.35 ± 0.17 ms | 5.69% ± 0.07% |
| simple_threshold | -18.56 ± 0.13 | 100.56 ± 0.12 ms | 130.56 ± 0.11 ms | 12.58% ± 0.09% |
| greedy_min_energy | -20.58 ± 0.04 | 111.39 ± 0.01 ms | 178.01 ± 0.26 ms | 24.19% ± 0.19% |
| random_policy | -27.42 ± 0.15 | 146.55 ± 0.88 ms | 209.27 ± 0.71 ms | 56.26% ± 0.61% |

*Source: `experiments/results_summary.json`*

**Stability note:** PPO results are highly consistent across seeds — std on SLA violation rate is only ±0.02 pp across the three seeds, demonstrating that the learned policy is not sensitive to environment initialisation.

### 4.3 Key findings (in the simulation setting)

- **SLA violation rate — PPO: 0.10% ± 0.02%** vs greedy_min_latency: 5.69% ± 0.07%.
  - **48× reduction** vs `greedy_min_latency` (seed=0 values: 5.79% ÷ 0.12% = 48×). Comparison against `weighted_heuristic` (the strongest hand-crafted baseline) gives a similar 6.89% vs 0.10% ratio.
- **Latency stability:** P95 latency bounded at **107.61 ms** (seed=0) / **107.18 ± 0.42 ms** (multi-seed) — consistently below the 120 ms SLA threshold. The agent reliably avoids high-load edge nodes in these scenarios.
- **Latency-energy trade-off:** In this simulation setup, PPO achieves a lower average latency than the latency-greedy strategy (93.6 ms vs 98.35 ms) while also using less energy (0.7225 vs 0.7280 Energy/Mbps), indicating a more favourable trade-off across both objectives.
- **Robustness:** Results are stable across three seeds. Per-seed details are in `experiments/results_summary.json`.

---

## 5. Plots

### 5.1 Episode Reward Comparison

![Episode Rewards](../experiments/plots_reward.png)

The learned policy (`rl_ppo`) achieves consistently higher episode rewards compared to all baselines in the simulation testing environment, with substantially fewer SLA penalty events.

### 5.2 Latency vs Energy Trade-off

![Trade-off](../experiments/plots_tradeoff.png)

The scatter shows all policies in latency × energy space. The ideal operating region is the bottom-left corner (low latency, low energy). PPO achieves the lowest observed latency among all policies in this simulation setup while staying near the group on energy, indicating a more favourable latency-energy trade-off compared to the heuristic baselines.

---

## 6. System Components

| Component | Command | Port |
|-----------|---------|------|
| Simulator smoke test | `python -m greenedge.simulator.smoke_test` | — |
| Training | `python -m greenedge.rl.train --algo ppo --steps 500000` | — |
| Evaluation (single seed) | `python -m greenedge.rl.evaluate --episodes 200 --seed 0` | — |
| Evaluation (multi-seed) | `python -m greenedge.rl.evaluate --episodes 200 --seeds 0 42 123` | — |
| API | `python -m greenedge.api.main` | 8000 |
| Dashboard | `streamlit run greenedge/dashboard/app.py` | 8501 |
| Full pipeline | `make all` | — |

---

## 7. Limitations

- **Simulation-only:** All results come from a Gymnasium simulation. No real 5G hardware or field measurements were used. Latency and energy models are physics-inspired abstractions, not empirical measurements.
- **Simplified dynamics:** The simulation assumes continuous state transitions with Gaussian noise. Real edge nodes exhibit bursty load patterns, OS-level jitter, and hardware thermal effects not captured here.
- **Static topology:** Fixed 2-edge + 1-cloud topology. Dynamic topologies (node joins/leaves) are not modelled.
- **No fault injection:** Scenarios with node failure or severe link degradation have not been systematically tested.
- **Observation synchronicity:** Telemetry is assumed instantaneous; real systems have measurement delays.

---

## 8. Conclusion

In the simulation setting, PPO-based RL consistently improves over all tested heuristics on the joint latency + energy + SLA objective. The trained agent achieves a **0.10% ± 0.02% SLA violation rate** (multi-seed mean ± std), compared to 5.69% ± 0.07% for `greedy_min_latency` — the strongest single-metric heuristic. On seed=0, this corresponds to a **48× reduction** (5.79% ÷ 0.12% = 48.25). The agent keeps P95 latency reliably below the 120 ms SLA threshold and achieves a more favourable latency-energy trade-off than latency-greedy approaches in these scenarios.

These results are reproducible from `experiments/policy.zip` and the evaluation scripts. The next step is shadow-mode deployment to validate the policy on real telemetry before production rollout.
