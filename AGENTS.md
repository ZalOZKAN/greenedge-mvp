# GreenEdge-5G - Agent Instructions

## 0) Goal (MVP outcome)
Build a working MVP that demonstrates an RL-based decision engine for routing workload to:
- edge-a
- edge-b
- cloud

The MVP must produce:
1) A runnable simulator (Gymnasium environment)
2) A trained policy (SB3 PPO or DQN) and baselines
3) An evaluation report (JSON + simple charts)
4) A dashboard (Streamlit) that shows KPI trade-offs live
5) A simple API (FastAPI) that returns decisions for a given observation

Primary audience: jury and non-technical reviewers.
Priority: stable demo + clear KPI visuals.

## 1) Repo structure (do not rename existing folders unless asked)
Current repo may contain old folders (part1/part2). Do not delete them.
Create new work under:

/greenedge
  /simulator        (Gym env + reward + scenario generation)
  /rl               (training + evaluation + baselines)
  /api              (FastAPI endpoints)
  /dashboard        (Streamlit UI)
/experiments        (results.json, plots, run logs)
/docs              (short report, figures, screenshots)

## 2) Definition of Done (acceptance)
The following commands must work on Windows (PowerShell) after venv install:

- python -m greenedge.simulator.smoke_test
  Prints 1 sample observation and 3 sample decisions with latency/energy numbers.

- python -m greenedge.rl.train --algo ppo --steps 20000
  Saves a policy file under /experiments (policy.zip or policy.pt).

- python -m greenedge.rl.evaluate --episodes 200
  Produces:
  /experiments/results.json
  /experiments/plots_reward.png
  /experiments/plots_tradeoff.png
  Metrics must include: avg_latency, p95_latency, energy_per_mbps, sla_violation_rate.

- python -m greenedge.api.main
  Starts FastAPI on localhost and exposes:
  POST /decision (takes obs[6], returns action + confidence + predicted_kpis)
  GET /health

- streamlit run greenedge/dashboard/app.py
  Shows:
  - KPI cards (latency, p95, energy/Mbps, SLA)
  - A live chart comparing RL vs baseline
  - A table showing last N decisions

## 3) Modeling rules (keep it simple and explainable)
Observation vector (size 6):
[ cpu_a, cpu_b, q_a, q_b, link_q, energy_price ]

Actions:
0 = route to edge-a
1 = route to edge-b
2 = route to cloud

Reward:
reward = -(alpha * energy + beta * latency + gamma * sla_penalty)
Provide default alpha/beta/gamma and allow changing via config.

Baselines required:
- greedy_min_latency
- greedy_min_energy
- simple_threshold (if cpu_a high then edge-b/cloud)

Confidence / safety:
Return a confidence score (0-1).
If confidence < threshold, fall back to baseline_min_latency.

## 4) Engineering rules
- Keep code readable and short. Prefer small functions.
- Add type hints in public functions.
- No heavy dependencies beyond: numpy, gymnasium, stable-baselines3, torch, fastapi, pydantic, streamlit, matplotlib/plotly.
- Every module must have a minimal __main__ runnable entry or a CLI.

## 5) Documentation outputs
Under /docs produce:
- demo_steps.md (how to run demo end-to-end)
- figures/ (screenshots, plots)
- mvp_report.md (1-2 pages, KPI table + 2 plots + short explanation)

## 6) Agent workflow (how to work)
Work in small increments:
Step A: implement simulator + smoke test
Step B: add SB3 training + save policy
Step C: add evaluation + plots + results.json
Step D: add FastAPI with typed request/response
Step E: add Streamlit dashboard reading results.json and showing live run
Step F: write docs and demo instructions

After each step:
- run the corresponding command
- fix errors before moving on
- commit with a short message

## 7) What NOT to do
- Do not add ns-3 or Mininet dependencies in MVP.
- Do not implement RL from scratch unless explicitly requested.
- Do not restructure the entire repo.
