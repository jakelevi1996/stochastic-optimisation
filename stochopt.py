import numpy as np

import objectives as o

class Solution():
    def __init__(self, location, fitness):
        self.location = location
        self.fitness = fitness

class TabuSearch():
    def __init__(self): pass
    
    def new_base_point(self, x, f=None):
        # Store location
        self.x_list.append(x)
        # If the fitness hasn't already been evaluated, then do so:
        if f is None:
            f = self.objective(x)
            self.n_evals += 1
            self.f_list.append(f)
        # Add to STM
        self.stm.append(x)
        # If the STM has exceeded max length, remove earliest item:
        while len(self.stm) > self.n_stm: self.stm.pop(0)
        # Add to MTM if among best solutions found so far
        # ...

        return f
        
    def diversify(self):
        # Find new random x-value
        x = np.random.uniform(self.x_min, self.x_max, self.ndims)
        # Add as new base point
        f = self.new_base_point(x)

        return x, f

    def local_search(self, x_initial):
        # # Make copy of most recent location
        # x = self.x_list[-1].copy()
        # Initialise lists to store perturbed values and their fitnesses
        x_list, f_list = [], []
        # Iterate through each dimension of x
        for d in range(self.ndims):
            # Try positive and negative perturbations
            for sign in [+1.0, -1.0]:
                # Make copy of current location and add perturbation
                xp = x_initial.copy()
                # print("Perturbing")
                xp[d] += sign * self.delta
                # Make sure the perturbed x-value is within limits
                xp[xp > self.x_max] = self.x_max
                xp[xp < self.x_min] = self.x_min
                # If x is tabu, don't evaluate it
                # print(xp, self.stm, any((xm == xp).all() for xm in self.stm))
                if not any((xp == xm).all() for xm in self.stm):
                    # Store and evaluate
                    x_list.append(xp)
                    f_list.append(self.objective(xp))
                    self.n_evals += 1
                # else: print(xp, "alreadt in stm")
        
        # Choose best allowed move
        f_best = min(f_list)
        x_best = x_list[f_list.index(f_best)]
        self.new_base_point(x_best, f_best)
        # Record all objective function evaluations:
        self.f_list.extend(f_list)

        return x_best, f_best

    def minimise(
        self, objective=o.schwefel, max_evals=10000, ndims=5, random_seed=0,
        n_stm=7, n_mtm=4, delta_initial=10, delta_reduction_factor=0.3,
        x_min=-500, x_max=500
    ):
        # Assign basic attributes for minimisation routine
        self.objective = objective
        self.max_evals = max_evals
        self.ndims = ndims
        self.n_stm, self.n_mtm = n_stm, n_mtm
        self.x_min, self.x_max = x_min, x_max
        self.delta = delta_initial
        np.random.seed(random_seed)
        # Find initial solution
        self.x_list, self.f_list, self.stm = [], [], []
        self.n_evals = 0
        x_new, f_new = self.diversify()
        # Begin main loop
        while self.n_evals <= max_evals:
            print("Local search")
            x_old, f_old = x_new, f_new
            x_new, f_new = self.local_search(x_old)
            if f_new < f_old:
                print("New value {:.5} -> pattern move".format(f_new))
            else: print("New value {:.5} -> Hill climb".format(f_new))
            # Pattern move...
            # Intensify...
            # Diversify...

if __name__ == "__main__":
    ts = TabuSearch()
    ts.minimise(max_evals=50, ndims=2)
    print("\nx list:")
    for x in ts.x_list: print(x)
    print("\nf list:")
    for f in ts.f_list: print(f)