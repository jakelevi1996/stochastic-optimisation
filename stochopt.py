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
        while len(self.stm) > self.N_STM: self.stm.pop(0)
        # Check if x is in MTM:
        if not self.x_in_mtm(x):
            self.mtm_x.append(x)
            self.mtm_f.append(f)
            # If the MTM has exceeded max length, remove least fit items:
            while check_reduce(len(self.mtm_x), len(self.mtm_f)) > self.N_MTM:
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
        if check_reduce(len(x_list), len(f_list)) > 0:
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

        return x, f

    def minimise(
        self, objective=o.schwefel, max_evals=10000, ndims=5, random_seed=0,
        n_stm=7, n_mtm=4, delta_initial=10, delta_reduction_factor=0.3,
        intensify_counter=12, diversify_counter=17, reduce_counter=25,
        x_min=-500, x_max=500, min_reduction=1.0
    ):
        # Set random seed and objective function
        np.random.seed(random_seed)
        self.objective = objective
        # Assign constant attributes for minimisation routine
        self.MAX_EVALS = max_evals
        self.N_DIMS = ndims
        self.N_STM, self.N_MTM = n_stm, n_mtm
        self.X_MIN, self.X_MAX = x_min, x_max
        # Initialise variable attributes for minimisation routine
        self.x_list, self.f_list,  = [], []
        self.x_diverse, self.f_diverse = [], []
        self.stm, self.mtm_x, self.mtm_f = [], [], []
        self.delta, self.n_evals = delta_initial, 0
        # Find initial solution
        x_new, f_new = self.diversify()
        # Begin main loop
        counter = 0
        max_counter = max(intensify_counter, diversify_counter, reduce_counter)
        while self.n_evals <= self.MAX_EVALS:
            # Perform local search
            x_old, f_old = x_new, f_new
            x_new, f_new = self.local_search(x_old, f_old)
            print("\nEval {}: Local search {:.4} -> {:.4}".format(self.n_evals, f_old, f_new))
            # Make pattern moves if the fitness is improving
            if f_new < f_old:
                dx, pm_ok = x_new - x_old, True
                while all([
                    f_new < f_old, pm_ok, self.n_evals <= self.MAX_EVALS
                ]):
                    x_old, f_old = x_new, f_new
                    x_new, f_new, pm_ok = self.pattern_move(x_old, f_old, dx)
                    print("Pattern move {:.4} -> {:.4}".format(f_old, f_new))
            else:
                counter += 1
                print("Counter incremented to", counter)
            # Intensify search if progress is not made
            if counter == intensify_counter:
                x_old, f_old = x_new, f_new
                x_new, f_new = self.intensify()
                counter += 1
                print("Intensify {:.4} -> {:.4}".format(f_old, f_new))
                # sleep(1)
            # Diversify search if progress is not made
            if counter == diversify_counter:
                x_old, f_old = x_new, f_new
                x_new, f_new = self.diversify()
                counter += 1
                print("Diversify {:.4} -> {:.4}".format(f_old, f_new))
                # sleep(1)
            # Reduce step size if progress is not made
            if counter == reduce_counter:
                self.delta = delta_reduction_factor * self.delta
                counter += 1
                print("Reducing step size")
                # sleep(1)
            if counter >= max_counter: counter = 0

            # Diversify...
            # TEMP: why does the program hang?
            # if all(x_new == x_old):
            #     x_old, f_old = x_new, f_new
            #     x_new, f_new = self.diversify()
            #     print("Diversifying")
        self.f_list = self.f_list[:max_evals]

if __name__ == "__main__":
    ts = TabuSearch()
    # ts.minimise(max_evals=10000, ndims=2, n_stm=7)
    # ts.minimise(max_evals=300, ndims=2, n_stm=7)
    ts.minimise(max_evals=3000, ndims=2, n_stm=300)
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
