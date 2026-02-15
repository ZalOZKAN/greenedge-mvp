# GreenEdge-5G — Demo Steps

End-to-end guide to run the full MVP demo on Windows (PowerShell).

---

## 0. Prerequisites

```powershell
# Python 3.10+ required
python --version

# Install dependencies (one-time)
pip install gymnasium numpy stable-baselines3 torch fastapi uvicorn pydantic streamlit plotly pandas matplotlib
```

---

## 1. Smoke Test  (Step A)

Verify the simulator works:

```powershell
python -m greenedge.simulator.smoke_test
```

**Expected:** 1 sample observation + 3 decisions (edge-a / edge-b / cloud) with latency, energy, and reward values.

---

## 2. Train RL Policy  (Step B)

Train a PPO agent for 20 000 timesteps:

```powershell
python -m greenedge.rl.train --algo ppo --steps 20000
```

**Expected:** Training logs printed → `experiments/policy.zip` saved (~20 seconds).

You can also train with DQN:

```powershell
python -m greenedge.rl.train --algo dqn --steps 20000
```

---

## 3. Evaluate  (Step C)

Run 200-episode evaluation comparing RL vs 3 baselines:

```powershell
python -m greenedge.rl.evaluate --episodes 200
```

**Expected outputs:**

| File | Description |
|------|-------------|
| `experiments/results.json` | Per-policy KPI metrics |
| `experiments/plots_reward.png` | Episode reward comparison chart |
| `experiments/plots_tradeoff.png` | Latency vs Energy scatter plot |

A summary table is printed to the terminal.

---

## 4. API Server  (Step D)

Start the FastAPI decision service:

```powershell
python -m greenedge.api.main
```

**Endpoints** (http://localhost:8000):

- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `POST /decision` → action + confidence + predicted KPIs
- Swagger docs → http://localhost:8000/docs

**Test with PowerShell:**

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Decision request
$body = '{"obs": [0.30, 0.52, 0.19, 0.21, 0.75, 0.30]}'
Invoke-RestMethod -Uri "http://localhost:8000/decision" -Method Post -Body $body -ContentType "application/json"
```

Press `Ctrl+C` to stop the server when done.

---

## 5. Dashboard  (Step E)

Launch the Streamlit dashboard:

```powershell
streamlit run greenedge/dashboard/app.py
```

**Opens at:** http://localhost:8501

**Dashboard sections:**
- KPI cards (latency, p95, energy, SLA violation %)
- Policy comparison table
- Latency vs Energy trade-off scatter (interactive)
- Live episode simulation (choose policy, run, see step-by-step chart)
- RL vs Baseline cumulative reward comparison

---

## Quick Full Demo (copy-paste)

```powershell
# 1. Smoke test
python -m greenedge.simulator.smoke_test

# 2. Train
python -m greenedge.rl.train --algo ppo --steps 20000

# 3. Evaluate
python -m greenedge.rl.evaluate --episodes 200

# 4. API (background)
Start-Process python -ArgumentList "-m greenedge.api.main"

# 5. Dashboard
streamlit run greenedge/dashboard/app.py
```
