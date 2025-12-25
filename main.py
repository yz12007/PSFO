import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opfunu
import random
import time
import warnings
import sys

# Ignore numerical warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PSFO Core Algorithm (v4.0 Final)
# ==========================================

class SearchAgent:
    """Represents a single candidate solution in the population"""
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.fitness = float('inf')
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.past_fitness = []

class PSFO_v4_Final:
    def __init__(self, obj_func, dim, bounds, pop_size=50, max_iter=1000):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        # Renamed 'pandas' to 'population'
        self.population = [SearchAgent(dim, bounds) for _ in range(pop_size)]
        self.global_best_pos = None
        self.global_best_fit = float('inf')
        self.history = []

    def check_bounds(self, position):
        """Strict boundary clipping to ensure stability on multimodal functions like F10"""
        return np.clip(position, self.bounds[0], self.bounds[1])

    def elite_local_search(self, current_iter):
        """Elite Local Search: Gaussian refinement with non-linear decay"""
        progress = current_iter / self.max_iter
        # Quartic decay (power of 4), step size becomes extremely small in later stages
        decay = (1 - progress) ** 4 
        scale = (self.bounds[1] - self.bounds[0]) * 0.01 * decay
        
        # Attempt 3 refinement trials
        for _ in range(3): 
            candidate_pos = self.global_best_pos + np.random.normal(0, scale, self.dim)
            candidate_pos = self.check_bounds(candidate_pos)
            candidate_fit = self.obj_func(candidate_pos)
            
            if candidate_fit < self.global_best_fit:
                self.global_best_fit = candidate_fit
                self.global_best_pos = candidate_pos
                # Sync to the first agent to prevent loss of the best solution
                self.population[0].position = candidate_pos.copy()
                self.population[0].fitness = candidate_fit

    def run(self):
        # --- Initialization & Evaluation ---
        for agent in self.population:
            fit = self.obj_func(agent.position)
            agent.fitness = fit
            agent.best_fitness = fit
            if fit < self.global_best_fit:
                self.global_best_fit = fit
                self.global_best_pos = agent.position.copy()
        
        self.history.append(self.global_best_fit)

        # --- Main Iteration Loop ---
        for t in range(self.max_iter):
            # 1. Phase Division and Parameter Decay
            progress = t / self.max_iter
            if progress < 0.33: 
                phase = "PHASE_I"
            elif progress < 0.66: 
                phase = "PHASE_II"
            else: 
                phase = "PHASE_III"

            # Guidance Factor (Cubic decay)
            alpha = 2.0 * (1 - progress) ** 3
            # Stochastic Variance (Environmental Noise)
            sigma = (self.bounds[1] - self.bounds[0]) * 0.02 * (1 - progress)

            for i, agent in enumerate(self.population):
                curr = agent.position
                new_pos = curr.copy()

                # --- 2. Multi-phase Strategies ---
                if phase == "PHASE_I":
                    # Phase I: Rapid Subspace Reduction (Linear Guidance Strategy)
                    # Exploration focus
                    r = np.random.random()
                    new_pos = curr + r * (self.global_best_pos - curr) * alpha
                
                elif phase == "PHASE_II":
                    # Phase II: Local Stabilization (Triangular Consensus Mechanism)
                    # Exploitation focus
                    neighbor_idx = (i + 1) % self.pop_size
                    neighbor_pos = self.population[neighbor_idx].position
                    new_pos = (curr + self.global_best_pos + neighbor_pos) / 3.0
                    # Add decaying Gaussian noise
                    new_pos += np.random.normal(0, sigma, self.dim)

                elif phase == "PHASE_III":
                    # Phase III: Diversity Re-injection (Hybrid Mutation Strategy)
                    # Stagnation Avoidance
                    if np.random.random() < 0.5:
                        # Strategy A: Cauchy Perturbation (Targeting multimodal/separable functions)
                        # Randomly select one dimension for Cauchy jump
                        dim_idx = np.random.randint(0, self.dim)
                        bound_width = self.bounds[1] - self.bounds[0]
                        # Heavy-tailed Cauchy distribution to maintain probability of escaping local optima
                        cauchy_step = np.random.standard_cauchy() * bound_width * 0.05 * (1-progress)
                        new_pos = self.global_best_pos.copy()
                        new_pos[dim_idx] += cauchy_step
                    else:
                        # Strategy B: Differential Evolution (Targeting rotated functions e.g., F1)
                        # Move using population distribution vectors
                        idxs = [idx for idx in range(self.pop_size) if idx != i]
                        r1, r2 = np.random.choice(idxs, 2, replace=False)
                        vec_diff = self.population[r1].position - self.population[r2].position
                        F = 0.5 # Differential Factor
                        new_pos = self.global_best_pos + F * vec_diff

                # 3. Boundary Constraint Check
                new_pos = self.check_bounds(new_pos)
                new_fit = self.obj_func(new_pos)
                
                # --- 4. Adaptive Re-initialization (formerly Fast Digestion) ---
                improvement = 0
                if len(agent.past_fitness) > 0:
                    prev = agent.past_fitness[-1]
                    if abs(prev) > 1e-15: 
                        improvement = (prev - new_fit) / abs(prev)
                
                # Stagnation Monitor: If stagnated and not the global best
                if improvement < 1e-8 and new_fit > self.global_best_fit:
                     # Gaussian Reset: Re-spawn around the global best
                     reset_sigma = (self.bounds[1] - self.bounds[0]) * 0.05 * (1 - progress)
                     new_pos = self.global_best_pos + np.random.normal(0, reset_sigma, self.dim)
                     new_pos = self.check_bounds(new_pos)
                     new_fit = self.obj_func(new_pos)

                # Update individual status
                agent.position = new_pos
                agent.fitness = new_fit
                agent.past_fitness.append(new_fit)

                if new_fit < agent.best_fitness:
                    agent.best_fitness = new_fit
                    agent.best_position = new_pos.copy()
                
                if new_fit < self.global_best_fit:
                    self.global_best_fit = new_fit
                    self.global_best_pos = new_pos.copy()
            
            # --- 5. Elite Local Search (Always Active) ---
            self.elite_local_search(t)
            self.history.append(self.global_best_fit)
            
        return self.global_best_fit, self.history
