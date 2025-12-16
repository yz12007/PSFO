# PSFO
# Panda Seasonal Foraging Optimizer (PSFO)

This repository contains the source code for the paper: **"Panda Seasonal Foraging Optimizer: An Effective Meta-Heuristic for Solving Constrained Engineering Design Problems"**.

## 🐼 Overview

Panda Seasonal Foraging Optimizer (PSFO) is a nature-inspired meta-heuristic algorithm that mimics the seasonal behavioral plasticity of giant pandas. It features a multi-phase temporal evolution mechanism to handle complex optimization landscapes:

* **Spring Phase (Linear Guidance):** For rapid exploration.
* **Autumn Phase (Triangular Consensus):** For local exploitation and stabilization.
* **Winter Phase (Hybrid Mutation):** Utilizing Cauchy flight and Differential Evolution to escape local optima.
* **Fast Digestion Mechanism:** A resource recycling strategy to counter stagnation in high-dimensional spaces.

## ⚙️ Requirements

* Python 3.8+
* NumPy
* Pandas
* Opfunu (for CEC benchmark functions)

Install dependencies via:
```bash
pip install -r requirements.txt
