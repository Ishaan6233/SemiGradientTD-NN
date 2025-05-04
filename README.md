# Semi-gradient TD with a Neural Network

This project implements **semi-gradient Temporal Difference (TD) learning** using a **neural network function approximator** to solve a **policy evaluation** problem on a 500-state Random Walk environment. This is a reimplementation of an assignment from [Reinforcement Learning: Function Approximation](https://www.deepblueai.com), part of the Deep Reinforcement Learning specialization.

## Overview

The objective is to approximate the true state-value function of a fixed policy using:

- A neural network with one hidden layer of 100 ReLU units.
- Semi-gradient TD(0) learning.
- The **Adam optimizer** for updating the network's weights.

The results are compared to a traditional linear function approximator using **tile coding**, demonstrating the trade-offs in sample efficiency and representational power.

---

## Files

- `agent.py` – Implements the TD agent using a neural network.
- `optimizer.py` – Contains both the `SGD` and `Adam` optimizers.
- `randomwalk_environment.py` – Defines the 500-state Random Walk environment.
- `rl_glue.py` – Lightweight RL-Glue framework to run agents/environments.
- `experiments/` – Runs experiments, tracks RMSVE, and plots learning curves.
- `plot_script.py` – Helper script to visualize results.
- `data/` – Contains `true_V.npy` and `state_distribution.npy` used for RMSVE calculation.
- `results/` – Stores generated plots and learning metrics.

---

## Key Concepts

- **Function Approximation**: Replacing lookup tables with parameterized function approximators (neural nets).
- **Semi-gradient TD(0)**: A bootstrapping method using temporal difference updates.
- **One-hot Encoding**: States are represented as sparse vectors to avoid introducing unintended similarity.
- **Adam Optimizer**: Used instead of vanilla SGD for adaptive learning rate and faster convergence.

---

## How to Run

```
# Set up environment (if using a notebook or local env)
pip install numpy matplotlib tqdm
Then, run the notebook or Python scripts to execute: python experiments/run_td_nn_experiment.py
This will:
- Train the TD agent across multiple runs and episodes.
- Track and save the RMSVE every 10 episodes.
- Generate plots for value approximation and learning curves.
```

## Results Summary
- The neural network approximates the true value function well but requires more samples than tile coding.
- Tile coding converges faster, while the neural network eventually generalizes better with enough data.
- Proper initialization and optimizer choice (Adam) are crucial for stability.

## Acknowledgements
This project was adapted from an assignment in the Reinforcement Learning Specialization by the University of Alberta, hosted on Coursera.
