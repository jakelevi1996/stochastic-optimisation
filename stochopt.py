import numpy as np
import logging
from time import sleep

import objectives as o
import utils as u
import plotting


class Minimiser():
    
    def evaluate_objective(self, x):
        # Wrapper for evaluating and recording the objective function
        f = self.objective(x)
        self.f_list.append(f)
        self.n_evals += 1
        # Check to see if this fitness is the best so far
        if f < self.f_best and self.n_evals <= self.MAX_EVALS:
            self.f_best = f
            self.x_best = x
            logging.info("nEval {}: New PB = {:.4}".format(self.n_evals, f))
        
        return f
    
    def minimise(
        self, objective=o.schwefel, random_seed=0, max_evals=10000
    ):
        # Set random seed and objective function
        np.random.seed(random_seed)
        self.objective = objective
        # Assign constant attributes
        self.MAX_EVALS = max_evals
        # Initialise variable attributes
        self.x_list, self.f_list,  = [], []
        self.n_evals = 0
        self.x_best, self.f_best = None, np.inf
        
        raise NotImplementedError


class TabuSearch(Minimiser):
    def __init__(self): pass
    
    def reset_memory(self):
        # Clear short and medium term memory and reset delta
        self.stm, self.mtm_x, self.mtm_f = [], [], []
        self.delta = self.DELTA_INITIAL

    # Check if x is in the short-term memory
    def x_in_stm(self, x): return any((x == xm).all() for xm in self.stm)
    # Check if x is in the medium-term memory
    def x_in_mtm(self, x): return any((x == xm).all() for xm in self.mtm_x)

    def new_base_point(self, x, f=None):
        # Store location
        self.x_list.append(x)
        # If the fitness hasn't already been evaluated, then do so:
        if f is None: f = self.evaluate_objective(x)
        # Add to STM
        self.stm.append(x)
        # If the STM has exceeded max length, remove earliest items:
        while len(self.stm) > self.N_STM: self.stm.pop(0)
        # Check if x is in MTM:
        if not self.x_in_mtm(x):
            self.mtm_x.append(x)
            self.mtm_f.append(f)
            # If the MTM has exceeded max length, remove least fit items:
            while u.reduce(len(self.mtm_x), len(self.mtm_f)) > self.N_MTM:
                i_worst = self.mtm_f.index(max(self.mtm_f))
                self.mtm_x.pop(i_worst)
                self.mtm_f.pop(i_worst)

        return f
    
    def local_search(self, x_initial, f_initial):
        # Initialise lists to store perturbed values and their fitnesses
        x_list, f_list = [], []
        # Iterate through each dimension of x
        for d in range(self.N_DIMS):
            # Try positive and negative perturbations
            for sign in [+1.0, -1.0]:
                # Make copy of current location and add perturbation
                xp = x_initial.copy()
                xp[d] += sign * self.delta
                # Make sure the perturbed x-value is within limits
                xp[xp > self.X_MAX] = self.X_MAX
                xp[xp < self.X_MIN] = self.X_MIN
                # Check if x is tabu
                if not self.x_in_stm(xp):
                    # Store and evaluate
                    x_list.append(xp)
                    f_list.append(self.evaluate_objective(xp))
        
        # If there are any non-tabu moves available:
        if u.reduce(len(x_list), len(f_list)) > 0:
            # Choose best allowed move:
            f_best = min(f_list)
            x_best = x_list[f_list.index(f_best)]
            # Record new base-point:
            self.new_base_point(x_best, f_best)
        # All moves are tabu; just return initial point
        else: x_best, f_best = x_initial, f_initial

        return x_best, f_best

    def pattern_move(self, x_old, f_old, dx):
        # Find new x value in feasible set
        x_new = x_old + dx
        x_new[x_new > self.X_MAX] = self.X_MAX
        x_new[x_new < self.X_MIN] = self.X_MIN
        # Check if new location is in STM
        if not self.x_in_stm(x_new):
            f_new = self.evaluate_objective(x_new)
            # If new fitness is better, record new base point
            if f_new < f_old:
                self.new_base_point(x_new, f_new)
                pattern_move_accepted = True
            # Otherwise discard the pattern move
            else: x_new, f_new, pattern_move_accepted = x_old, f_old, False
        # Pattern move is in STM; just return initial point
        else: x_new, f_new, pattern_move_accepted = x_old, f_old, False
        
        return x_new, f_new, pattern_move_accepted
    
    def intensify(self):
        # Find new intensified x-value
        x = np.mean(self.mtm_x, axis=0)
        # Add as new base point
        f = self.new_base_point(x)

        return x, f
        
    def diversify(self):
        # Find new random x-value
        x = np.random.uniform(self.X_MIN, self.X_MAX, self.N_DIMS)
        # Add as new base point
        f = self.new_base_point(x)
        self.x_diverse.append(x)
        self.f_diverse.append(f)
        # Clear short and medium term memory? Reset delta?
        self.reset_memory()

        return x, f

    def minimise(
        self, objective=o.schwefel, max_evals=10000, n_dims=5, random_seed=0,
        x_min=-500, x_max=500,
        # Performance args:
        n_stm=7, n_mtm=4, delta_initial=10, delta_reduction_factor=0.2,
        max_counter=25, min_reduction=1.0
    ):
        # Set random seed and objective function
        np.random.seed(random_seed)
        self.objective = objective
        # Assign constant attributes
        self.MAX_EVALS = max_evals
        self.N_DIMS = n_dims
        self.N_STM, self.N_MTM = n_stm, n_mtm
        self.DELTA_INITIAL = delta_initial
        self.X_MIN, self.X_MAX = x_min, x_max
        # Initialise variable attributes
        self.x_list, self.f_list,  = [], []
        self.x_diverse, self.f_diverse = [], []
        self.n_evals = 0
        self.reset_memory()
        self.x_best, self.f_best = None, np.inf
        # Find initial solution
        x_new, f_new = self.diversify()
        # Begin main loop
        counter = 0
        # max_counter = max(intensify_counter, diversify_counter, reduce_counter)
        f_last_reduction = f_new
        while self.n_evals <= self.MAX_EVALS:
            # Perform local search
            x_old, f_old = x_new, f_new
            x_new, f_new = self.local_search(x_old, f_old)
            u.log_local_search(self.n_evals, f_old, f_new)
            # Make pattern moves if the fitness is improving
            if f_new < f_old:
                dx, pm_ok = x_new - x_old, True
                while all([
                    f_new < f_old, pm_ok, self.n_evals <= self.MAX_EVALS
                ]):
                    x_old, f_old = x_new, f_new
                    x_new, f_new, pm_ok = self.pattern_move(x_old, f_old, dx)
                    u.log_pattern_move(f_old, f_new)
            # If previous local search didn't improve fitness, increase counter
            else:
                counter += 1
                u.log_counter(counter)
            # Intensify or diversify search if progress is not being made
            if counter >= max_counter:
                x_old, f_old = x_new, f_new
                x_new, f_new = self.intensify()
                self.delta = delta_reduction_factor * self.delta
                u.log_intensify(f_old, f_new)
                # If most recent intensification gave bad reduction in fitness
                if not f_last_reduction - f_new > min_reduction:
                    # Diversify and start again
                    u.log_diversify(f_last_reduction, f_new)
                    x_new, f_new = self.diversify()
                # Either way, reset baseline fitness and counter
                f_last_reduction = f_new
                counter = 0

        self.f_list = self.f_list[:max_evals]

        return self.x_best, self.f_best

class Particle():
    def __init__(self, n_dims, x_min, x_max, more_args="..."):
        # self.pbest = None
        self.x = None
        self.v = None
        self.f_best = np.inf
        self.x_best = None # (this one is actually none)
        # ...
    
    def move(self, gbest):
        self.v += None
        self.x += self.v
        return self.x
    
    def update_best(self, x, f):
        if f <= self.f_best: self.x_best, self.f_best = x, f

        

class ParticleSwarm(Minimiser):
    def minimise(
        self, objective=o.schwefel, max_evals=10000, n_dims=5, random_seed=0,
        x_min=-500, x_max=500,
        # Performance args:
        num_particles=100,
    ):
        # Set random seed and objective function
        np.random.seed(random_seed)
        self.objective = objective
        # Assign constant attributes
        self.MAX_EVALS = max_evals
        # Initialise variable attributes
        self.x_list, self.f_list,  = [], []
        self.n_evals = 0
        self.x_best, self.f_best = None, np.inf

        particle_list = [
            Particle(n_dims, x_min, x_max) for _ in range(num_particles)
        ]
        num_evals = 3 # not actually 3
        for _ in range(num_evals):
            # For each particle in the swarm
            for p in particle_list:
                x = p.move()
                f = self.evaluate_objective(x)
                p.update_best(x, f)
                # ...


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)
    ts = TabuSearch()
    # ts.minimise(max_evals=10000, n_dims=2, n_stm=7)
    # ts.minimise(max_evals=300, n_dims=2, n_stm=7)
    # print(ts.minimise(max_evals=3000, n_dims=2, n_stm=7))
    # print(ts.minimise(max_evals=10000, n_dims=2, n_stm=100, max_counter=100))
    print(ts.minimise(max_evals=10000, n_dims=2, n_stm=10, max_counter=10, random_seed=3))
    # print("\nx list:")
    # for x in ts.x_list: print(x)
    # print("\nf list:")
    # for f in ts.f_list: print(f)
    print("\nMTM x list:")
    for x in ts.mtm_x: print(x)
    print("\nMTM f list:")
    for f in ts.mtm_f: print(f)
    # plotting.plot_objective(
    #     filename="Schwefel tabu", x_list=ts.x_list,
    #     x_list_d=ts.x_diverse
    # )
    plotting.plot_fitness_history(
        ts.f_list, filename="Schwefel fitness local search diversify"
    )

    pso = ParticleSwarm()
    pso.minimise()
