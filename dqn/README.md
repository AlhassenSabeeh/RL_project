# 🤖 TurtleBot3 DQN — Reinforcement Learning with PyTorch

An optimized Deep Q-Network (DQN) implementation for TurtleBot3, featuring real-time training visualization and post-training performance analysis.

---

## 📋 Prerequisites

Before getting started, make sure you have the following installed:

**System Requirements**
- ROS 2 (Humble or Foxy)
- Gazebo Simulation Environment

**Python Dependencies**

```bash
pip install torch torchvision pandas matplotlib numpy pyqtgraph
```

---

## 🚀 Getting Started

The simulation requires **three separate terminals** running in parallel. Follow the steps below in order.

---

### Step 1 — Launch the Simulation

In your **first terminal**, start the Gazebo world:

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
```

> 💡 You can swap `stage1` for `stage2`, `stage3`, or `stage4` depending on the environment complexity you want.

---

### Step 2 — Start the Environment Node

In your **second terminal**, launch the environment node. This handles state calculation and reward signals:

```bash
python3 dqn_environment.py
```

---

### Step 3 — Run the Training Agent

In your **third terminal**, start the DQN training script:

```bash
# Usage: python3 dqn_agent.py [stage] [max_episodes]
python3 dqn_agent.py 1 1000
```

---

### Step 4 — Monitor Live Progress *(Optional)*

To watch reward and Q-value graphs update in real time, open a **fourth terminal**:

```bash
python3 result_graph.py
```

---

## 📊 Post-Training Analysis

Once training has completed (or after **200+ episodes**), generate a formal performance report:

```bash
python3 training_analysis.py
```

**This produces the following outputs:**

| Output | Description |
|---|---|
| `performance_report_DQN_Initial.png` | High-resolution plot of Success Rate, Total Reward, and Training Loss |
| `saved_model/training_log_stage1.csv` | Raw episode-by-episode training data |
| `saved_model/` | Trained model weights checkpoint |

---

## 📦 Deliverables Checklist

After running the simulation, please share the following for review:

- [ ] **CSV Log** — `saved_model/training_log_stage1.csv`
- [ ] **Performance Graph** — `performance_report_DQN_Initial.png`
- [ ] **Collision Rate Observation** — Notes on collision behavior during the first 50 episodes

---

## 🔧 Troubleshooting

**Resuming a previous training run**

Open `dqn_agent.py` and set the following flag to `True`:

```python
self.load_model = True
```

This will load from the last saved checkpoint instead of starting fresh.

---

**CSV log not generating**

Check two things:
1. Confirm the `saved_model/` directory exists in your working folder.
2. Verify the script has write permissions to that directory.

```bash
mkdir -p saved_model
```