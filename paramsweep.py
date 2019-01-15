import numpy as np
import logging
import matplotlib.pyplot as plt

import stochopt as so

def sweep_pso_v_decay(
    num_seeds=10, v_decay_list=np.linspace(0.1, 1.0, 10), figsize=[8, 6],
    max_evals=10000, n_dims=5,
    num_particles=100, p_inc=2.0, g_inc=1.0, v_max=10.0
):
    # Plot graph of fitness against v_decay
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(v_decay_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, v_decay in enumerate(v_decay_list):
            _, f = so.ParticleSwarm().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                v_decay=v_decay, num_particles=num_particles, p_inc=p_inc,
                g_inc=g_inc, v_max=v_max
            )
            print(f)
            f_array[i, j] = f
        plt.plot(v_decay_list, f_array[i], "bo", alpha=0.5)
    
    plt.plot(v_decay_list, f_array.mean(axis=0), "ro-")
    
    plt.grid(True)
    plt.xlabel(r"$v_{decay}$")
    plt.ylabel("Fitness")
    plt.title(r"Effect of $v_{decay}$ on fitness")
    plt.savefig("Images/PSO fitness v v_decay")

def sweep_pso_num_particles(
    num_seeds=10, num_particles_list=np.logspace(1, 3, 15), figsize=[8, 6],
    max_evals=10000, n_dims=5,
    v_decay=0.9, p_inc=2.0, g_inc=1.0, v_max=10.0
):
    # Plot graph of fitness against v_decay
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(num_particles_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, num_particles in enumerate(num_particles_list):
            num_particles = int(num_particles)
            _, f = so.ParticleSwarm().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                v_decay=v_decay, num_particles=num_particles, p_inc=p_inc,
                g_inc=g_inc, v_max=v_max
            )
            print(f)
            f_array[i, j] = f
        plt.semilogx(num_particles_list, f_array[i], "bo", alpha=0.5)
    
    plt.semilogx(num_particles_list, f_array.mean(axis=0), "ro-")
    
    plt.grid(True)
    plt.xlabel("Number of particles")
    plt.ylabel("Fitness")
    plt.title("Effect of number of particles on fitness")
    plt.savefig("Images/PSO fitness v number of particles")

def sweep_pso_p_inc(
    num_seeds=10, p_inc_list=np.linspace(0.6, 2.6, 11), figsize=[8, 6],
    max_evals=10000, n_dims=5,
    v_decay=0.9, num_particles=100, g_inc=1.0, v_max=10.0
):
    # Plot graph of fitness against v_decay
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(p_inc_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, p_inc in enumerate(p_inc_list):
            _, f = so.ParticleSwarm().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                v_decay=v_decay, num_particles=num_particles, p_inc=p_inc,
                g_inc=g_inc, v_max=v_max
            )
            print(f)
            f_array[i, j] = f
        plt.plot(p_inc_list, f_array[i], "bo", alpha=0.5)
    
    plt.plot(p_inc_list, f_array.mean(axis=0), "ro-")

    plt.grid(True)
    plt.xlabel(r"$p_{inc}$")
    plt.ylabel("Fitness")
    plt.title(r"Effect of $p_{inc}$ on fitness")
    plt.savefig("Images/PSO fitness v p_inc")

def sweep_pso_g_inc(
    num_seeds=10, g_inc_list=np.linspace(0.6, 2.6, 11), figsize=[8, 6],
    max_evals=10000, n_dims=5,
    v_decay=0.9, num_particles=100, p_inc=2.0, v_max=10.0
):
    # Plot graph of fitness against v_decay
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(g_inc_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, g_inc in enumerate(g_inc_list):
            _, f = so.ParticleSwarm().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                v_decay=v_decay, num_particles=num_particles, p_inc=p_inc,
                g_inc=g_inc, v_max=v_max
            )
            print(f)
            f_array[i, j] = f
        plt.plot(g_inc_list, f_array[i], "bo", alpha=0.5)
    
    plt.plot(g_inc_list, f_array.mean(axis=0), "ro-")
    
    plt.grid(True)
    plt.xlabel(r"$g_{inc}$")
    plt.ylabel("Fitness")
    plt.title(r"Effect of $g_{inc}$ on fitness")
    plt.savefig("Images/PSO fitness v g_inc")

def sweep_pso_v_max(
    num_seeds=10, v_max_list=np.logspace(0, 2, 11), figsize=[8, 6],
    max_evals=10000, n_dims=5,
    v_decay=0.9, num_particles=100, p_inc=2.0, g_inc=1.0
):
    # Plot graph of fitness against v_decay
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(v_max_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, v_max in enumerate(v_max_list):
            _, f = so.ParticleSwarm().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                v_decay=v_decay, num_particles=num_particles, p_inc=p_inc,
                g_inc=g_inc, v_max=v_max
            )
            print(f)
            f_array[i, j] = f
        plt.semilogx(v_max_list, f_array[i], "bo", alpha=0.5)
    
    plt.semilogx(v_max_list, f_array.mean(axis=0), "ro-")
    
    plt.grid(True)
    plt.xlabel(r"$v_{max}$")
    plt.ylabel("Fitness")
    plt.title(r"Effect of $v_{max}$ on fitness")
    plt.savefig("Images/PSO fitness v v_max")

def sweep_ts_max_counter(
    num_seeds=10, max_counter_list=np.logspace(0, 2, 11, dtype=np.int),
    figsize=[8, 6], max_evals=10000, n_dims=5,
    n_stm=7, n_mtm=4, delta_initial=10, delta_reduction_factor=0.2,
    min_reduction=1.0
):
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(max_counter_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, max_counter in enumerate(max_counter_list):
            _, f = so.TabuSearch().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                max_counter=max_counter, n_stm=n_stm, n_mtm=n_mtm,
                min_reduction=min_reduction, delta_initial=delta_initial,
                delta_reduction_factor=delta_reduction_factor,
            )
            print(f)
            f_array[i, j] = f
        plt.semilogx(max_counter_list, f_array[i], "bo", alpha=0.5)
    
    plt.semilogx(max_counter_list, f_array.mean(axis=0), "ro-")
    
    plt.grid(True)
    plt.xlabel("Counter limit")
    plt.ylabel("Fitness")
    plt.title("Effect of counter limit on fitness")
    plt.savefig("Images/TS fitness v max_counter")

def sweep_ts_n_stm(
    num_seeds=10, n_stm_list=np.logspace(0, 2, 11, dtype=np.int),
    figsize=[8, 6], max_evals=10000, n_dims=5,
    max_counter=10, n_mtm=4, delta_initial=10, delta_reduction_factor=0.2,
    min_reduction=1.0
):
    plt.figure(figsize=figsize)

    f_array = np.empty(shape=[num_seeds, len(n_stm_list)])
    for i, random_seed in enumerate(range(num_seeds)):
        for j, n_stm in enumerate(n_stm_list):
            _, f = so.TabuSearch().minimise(
                max_evals=max_evals, n_dims=n_dims, random_seed=random_seed,
                max_counter=max_counter, n_stm=n_stm, n_mtm=n_mtm,
                min_reduction=min_reduction, delta_initial=delta_initial,
                delta_reduction_factor=delta_reduction_factor,
            )
            print(f)
            f_array[i, j] = f
        plt.semilogx(n_stm_list, f_array[i], "bo", alpha=0.5)
    
    plt.semilogx(n_stm_list, f_array.mean(axis=0), "ro-")
    
    plt.grid(True)
    plt.xlabel("STM size")
    plt.ylabel("Fitness")
    plt.title("Effect of STM size on fitness")
    plt.savefig("Images/TS fitness v n_stm")



if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # sweep_pso_v_decay(v_max=10, g_inc=1.0)
    # sweep_pso_num_particles(v_max=10, g_inc=1.0)
    # sweep_pso_p_inc(v_max=10, g_inc=1.0)
    # sweep_pso_g_inc(v_max=10)
    # sweep_pso_v_max(g_inc=1.0)
    
    sweep_ts_max_counter()
    sweep_ts_n_stm()
