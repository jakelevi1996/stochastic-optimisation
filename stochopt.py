import numpy as np
from time import sleep

import objectives as o
import plotting

def check_reduce(*arg_list):
    arg_set = set(arg_list)
    assert len(arg_set) == 1
    return arg_set.pop()

class Solution():
    def __init__(self, location, fitness):
        self.location = location
        self.fitness = fitness

class TabuSearch():
    def __init__(self): pass
    
    def evaluate_objective(self, x):
        # Wrapper for evaluating and recording the objective function
        f = self.objective(x)
        self.f_list.append(f)
        self.n_evals += 1
        
        return f

    # Method for checking if an x value is in the short-term memory
    def x_in_stm(self, x): return any((x == xm).all() for xm in self.stm)
    # Method for checking if an x value is in the medium-term memory
    def x_in_mtm(self, x): return any((x == xm).all() for xm in self.mtm_x)

    def new_base_point(self, x, f=None):
        # Store location
        self.x_list.append(x)
        # If the fitness hasn't already been evaluated, then do so:
        if f is None: f = self.evaluate_objective(x)
        # Add to STM
        self.stm.append(x)
        # If the STM has exceeded max length, remove earliest items:
        while len(self.stm) > self.n_stm: self.stm.pop(0)
        # Check if x is in MTM:
        if not self.x_in_mtm(x):
            self.mtm_x.append(x)
            self.mtm_f.append(f)
            # If the MTM has exceeded max length, remove least fit items:
            while check_reduce(len(self.mtm_x), len(self.mtm_f)) > self.n_mtm:
                i_worst = self.mtm_f.index(max(self.mtm_f))
                self.mtm_x.pop(i_worst)
                self.mtm_f.pop(i_worst)

        return f
        
    def diversify(self):
        # Find new random x-value
        x = np.random.uniform(self.x_min, self.x_max, self.ndims)
        # Add as new base point
        f = self.new_base_point(x)
        self.x_diverse.append(x)
        self.f_diverse.append(f)

        return x, f
    

    def local_search(self, x_initial, f_initial):
        # Initialise lists to store perturbed values and their fitnesses
        x_list, f_list = [], []
        # Iterate through each dimension of x
        for d in range(self.ndims):
            # Try positive and negative perturbations
            for sign in [+1.0, -1.0]:
                # Make copy of current location and add perturbation
                xp = x_initial.copy()
                xp[d] += sign * self.delta
                # Make sure the perturbed x-value is within limits
                xp[xp > self.x_max] = self.x_max
                xp[xp < self.x_min] = self.x_min
                # Check if x is tabu
                if not self.x_in_stm(xp):
                    # Store and evaluate
                    x_list.append(xp)
                    f_list.append(self.evaluate_objective(xp))
        
        # If there are any non-tabu moves available:
        if check_reduce(len(x_list), len(f_list)) > 0:
            # Choose best allowed move:
            f_best = min(f_list)
            x_best = x_list[f_list.index(f_best)]
            # Record new base-point:
            self.new_base_point(x_best, f_best)
        else:
            # All moves are tabu; just return initial point
            print("All perturbations are tabu")
            x_best, f_best = x_initial, f_initial

        return x_best, f_best

    def pattern_move(self, x_old, f_old, dx):
        print("Pattern move")
        # Find new x value in feasible set
        x_new = x_old + dx
        x_new[x_new > self.x_max] = self.x_max
        x_new[x_new < self.x_min] = self.x_min

        # Check if new location is in in STM
        if not self.x_in_stm(x_new):
            pattern_location_allowed = True
            f_new = self.evaluate_objective(x_new)
            # If new fitness is better, record new base point
            if f_new < f_old: self.new_base_point(x_new, f_new)

            if f_new < f_old: print("New f {:.5} accepted".format(f_new))
            else: print("New f {:.5} rejected".format(f_new))
        else:
            # Pattern move is in STM; just return initial point
            print("Pattern move in STM")
            # sleep(1)
            pattern_location_allowed = False
            x_new, f_new = x_old, f_old
        
        return x_new, f_new, pattern_location_allowed

    def minimise(
        self, objective=o.schwefel, max_evals=10000, ndims=5, random_seed=0,
        n_stm=7, n_mtm=4, delta_initial=10, delta_reduction_factor=0.3,
        x_min=-500, x_max=500
    ):
        # Set random seed
        np.random.seed(random_seed)
        # Assign constant attributes for minimisation routine
        self.objective = objective
        self.max_evals = max_evals
        self.ndims = ndims
        self.n_stm, self.n_mtm = n_stm, n_mtm
        self.x_min, self.x_max = x_min, x_max
        self.delta = delta_initial
        # Initialise variable attributes for minimisation routine
        self.x_list, self.f_list,  = [], []
        self.x_diverse, self.f_diverse = [], []
        self.stm, self.mtm_x, self.mtm_f = [], [], []
        self.delta, self.n_evals = delta_initial, 0
        # Find initial solution
        x_new, f_new = self.diversify()
        # Begin main loop
        while self.n_evals <= max_evals:
            x_old, f_old = x_new, f_new
            print("\nLocal search")
            x_new, f_new = self.local_search(x_old, f_old)
            dx = x_new - x_old
            pm_ok = True
            if not f_new < f_old:
                print("Eval {}: New value {:.5} -> Hill climb".format(self.n_evals, f_new))
            while f_new < f_old and pm_ok:
                print("Eval {}: New value {:.5} -> pattern move".format(self.n_evals, f_new))
                x_old, f_old = x_new, f_new
                x_new, f_new, pm_ok = self.pattern_move(x_old, f_old, dx)
            # Pattern move...
            # Intensify...
            # Diversify...
            # TEMP: why does the program hang?
            if all(x_new == x_old):
                x_old, f_old = x_new, f_new
                x_new, f_new = self.diversify()
                print("Diversifying")
        self.f_list = self.f_list[:max_evals]

if __name__ == "__main__":
    ts = TabuSearch()
    ts.minimise(max_evals=10000, ndims=2, n_stm=7)
    # print("\nx list:")
    # for x in ts.x_list: print(x)
    # print("\nf list:")
    # for f in ts.f_list: print(f)
    print("\nMTM x list:")
    for x in ts.mtm_x: print(x)
    print("\nMTM f list:")
    for f in ts.mtm_f: print(f)
    plotting.plot_objective(
        filename="Schwefel local search", x_list=ts.x_list,
        x_list_d=ts.x_diverse
    )
    plotting.plot_fitness_history(
        ts.f_list, filename="Schwefel fitness local search diversify"
    )
