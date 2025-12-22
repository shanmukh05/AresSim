# AresSim - Quick Start Instructions

## Prerequisites

- **Node.js** 18+
- **Python** 3.9+
- **npm** or **yarn**

---

## Installation

### 1. Install React UI dependencies
```bash
npm install
```

### 2. Install Python RL environment
```bash
# Basic install
pip install -e .

# With training dependencies (recommended)
pip install -e ".[train]"
```

---

## Running the Application

### Local Mode (JavaScript simulation)
```bash
npm run dev
```
Open http://localhost:5173 and use controls to interact.

---

### Remote Mode (Python environment)

**Option A: Manual control server**
```bash
# Terminal 1: Start Python server
python -m gym_mars.server

# Terminal 2: Start React UI
npm run dev
```
Then switch to **Remote** mode in the UI.

**Option B: Demo with random actions**
```bash
# Terminal 1: Run demo
python scripts/demo_random.py

# Terminal 2: Start React UI
npm run dev
```
Switch to Remote, then click **RESUME SIM** to watch.

---

## Training RL Agents

### Train a PPO agent
```bash
# Quick training (100k steps)
python scripts/train.py --total-timesteps 100000

# Better exploration (recommended)
python scripts/train.py --total-timesteps 500000 --ent-coef 0.05
```

### Monitor training
```bash
tensorboard --logdir logs/
```

### Visualize trained agent
```bash
python scripts/run_with_viz.py models/mars_ppo_*/final_model.zip

# In another terminal
npm run dev
```
Switch to Remote mode and click **RESUME SIM**.

---

## Using as Gymnasium Environment

```python
import gymnasium as gym
import gym_mars

env = gym.make("gym_mars/MarsSurvival-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## Action Reference

| ID | Action |
|----|--------|
| 0 | Move North |
| 1 | Move South |
| 2 | Move East |
| 3 | Move West |
| 4 | Mine Ice |
| 5 | Use Oxygen Tank |
| 6 | Charge Battery |
