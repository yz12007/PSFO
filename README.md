# PSFO
# Panda Seasonal Foraging Optimizer (PSFO)

This repository contains the source code for the paper: **"Phased Search and Fine-tuning Optimizer: An Effective Meta-Heuristic for Solving Constrained Engineering Design Problems"**.

## 🐼 Overview

The framework incorporates three distinct mathematical phases:
(1) Phase I (Rapid Exploration) utilizes a linear guidance strategy for aggressive subspace reduction.
(2) Phase II (Local Stabilization) employs a triangular consensus mechanism to refine the population around promising regions.
(3) Phase III (Diversity Re-injection) features a hybrid mutation strategy (integrating Cauchy flight and Differential Evolution) to effectively escape local optima.

## ⚙️ Requirements

* Python 3.8+
* NumPy
* Pandas
* Opfunu (for CEC benchmark functions)

Install dependencies via:
```bash
pip install -r requirements.txt
