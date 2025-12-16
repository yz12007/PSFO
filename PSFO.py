import numpy as np

class Panda:
    """
    Individual agent in the PSFO algorithm.
    """
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.fitness = float('inf')
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.past_fitness = []

class PSFO:
    """
    Panda Seasonal Foraging Optimizer (PSFO)
    
    A bio-inspired meta-heuristic algorithm based on the seasonal behavioral 
    plasticity of giant pandas.
    
    Phases:
    1. Spring (Linear Guidance): Rapid exploration.
    2. Autumn (Triangular Consensus): Local stabilization.
    3. Winter (Hybrid Mutation): Anti-stagnation & diversity injection.
    + Fast Digestion: Resource recycling for inactive particles.
    """
    def __init__(self, obj_func, dim, bounds, pop_size=50, max_iter=1000):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pandas = [Panda(dim, bounds) for _ in range(pop_size)]
        self.global_best_pos = None
        self.global_best_fit = float('inf')
        
    def check_bounds(self, position):
        return np.clip(position, self.bounds[0], self.bounds[1])

    def elite_local_search(self, current_iter):
        """
        Performs fine-grained local search around the global best 
        using a quartic decay step size.
        """
        progress = current_iter / self.max_iter
        decay = (1 - progress) ** 4 
        scale = (self.bounds[1] - self.bounds[0]) * 0.01 * decay
        
        for _ in range(3): 
            candidate_pos = self.global_best_pos + np.random.normal(0, scale, self.dim)
            candidate_pos = self.check_bounds(candidate_pos)
            candidate_fit = self.obj_func(candidate_pos)
            
            if candidate_fit < self.global_best_fit:
                self.global_best_fit = candidate_fit
                self.global_best_pos = candidate_pos
                # Update the leader panda as well
                self.pandas[0].position = candidate_pos.copy()
                self.pandas[0].fitness = candidate_fit

    def run(self):
        # Initialization
        for panda in self.pandas:
            fit = self.obj_func(panda.position)
            panda.fitness = fit
            panda.best_fitness = fit
            if fit < self.global_best_fit:
                self.global_best_fit = fit
                self.global_best_pos = panda.position.copy()

        # Main Loop
        for t in range(self.max_iter):
            progress = t / self.max_iter
            
            # Determine Season (Phase)
            if progress < 0.33: 
                season = "SPRING" # Linear Guidance
            elif progress < 0.66: 
                season = "AUTUMN" # Triangular Consensus
            else: 
                season = "WINTER" # Mutation Strategy
            
            alpha = 2.0 * (1 - progress) ** 3
            sigma = (self.bounds[1] - self.bounds[0]) * 0.02 * (1 - progress)

            for i, panda in enumerate(self.pandas):
                curr = panda.position
                new_pos = curr.copy()
                
                # --- Phase Updates ---
                if season == "SPRING":
                    r = np.random.random()
                    new_pos = curr + r * (self.global_best_pos - curr) * alpha
                    
                elif season == "AUTUMN":
                    neighbor_idx = (i + 1) % self.pop_size
                    neighbor_pos = self.pandas[neighbor_idx].position
                    new_pos = (curr + self.global_best_pos + neighbor_pos) / 3.0
                    new_pos += np.random.normal(0, sigma, self.dim)
                    
                elif season == "WINTER":
                    if np.random.random() < 0.5:
                        # Strategy A: Cauchy Flight (Long-tail jumps)
                        dim_idx = np.random.randint(0, self.dim)
                        bound_width = self.bounds[1] - self.bounds[0]
                        cauchy_step = np.random.standard_cauchy() * bound_width * 0.05 * (1-progress)
                        new_pos = self.global_best_pos.copy()
                        new_pos[dim_idx] += cauchy_step
                    else:
                        # Strategy B: Differential Evolution (Scramble)
                        idxs = [idx for idx in range(self.pop_size) if idx != i]
                        r1, r2 = np.random.choice(idxs, 2, replace=False)
                        vec_diff = self.pandas[r1].position - self.pandas[r2].position
                        F = 0.5 
                        new_pos = self.global_best_pos + F * vec_diff
                
                # Boundary Check & Evaluation
                new_pos = self.check_bounds(new_pos)
                new_fit = self.obj_func(new_pos)
                
                # --- Fast Digestion (Resource Recycling Mechanism) ---
                # Rationale: Inactive particles are recycled to the promising region 
                # to increase effective sampling density.
                improvement = 0
                if len(panda.past_fitness) > 0:
                    prev = panda.past_fitness[-1]
                    if abs(prev) > 1e-15: improvement = (prev - new_fit) / abs(prev)
                
                if improvement < 1e-8 and new_fit > self.global_best_fit:
                      reset_sigma = (self.bounds[1] - self.bounds[0]) * 0.05 * (1 - progress)
                      new_pos = self.global_best_pos + np.random.normal(0, reset_sigma, self.dim)
                      new_pos = self.check_bounds(new_pos)
                      new_fit = self.obj_func(new_pos)

                # Update Individual
                panda.position = new_pos
                panda.fitness = new_fit
                panda.past_fitness.append(new_fit)
                
                if new_fit < panda.best_fitness:
                    panda.best_fitness = new_fit
                    panda.best_position = new_pos.copy()
                
                # Update Global Best
                if new_fit < self.global_best_fit:
                    self.global_best_fit = new_fit
                    self.global_best_pos = new_pos.copy()
            
            # Elite Local Search (End of Iteration)
            self.elite_local_search(t)
            
        return self.global_best_fit