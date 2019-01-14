import logging

def reduce(*arg_list):
    arg_set = set(arg_list)
    assert len(arg_set) == 1
    return arg_set.pop()

class Solution():
    def __init__(self, location, fitness):
        self.location = location
        self.fitness = fitness

def log_local_search(n_evals, f_old, f_new):
    logging.debug("\nEval {}:".format(n_evals))
    logging.debug("Local search {:.4} -> {:.4}".format(f_old, f_new))

def log_pattern_move(f_old, f_new):
    logging.debug("Pattern move {:.4} -> {:.4}".format(f_old, f_new))

def log_counter(counter):
    logging.debug("Counter -> {}".format(counter))

def log_intensify(f_old, f_new):
    logging.debug("Intensify {:.4} -> {:.4}".format(f_old, f_new))

def log_diversify(f_lr, f_new):
    logging.debug("Last intensification: {:.4} -> {:.4}".format(f_lr, f_new))
    logging.debug("Diversifying...")
