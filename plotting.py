import numpy as np
import matplotlib.pyplot as plt

import objectives as o

def plot_objective(
    objective=o.schwefel, x0lims=[-500, 500], x1lims=[-500, 500], n=200,
    filename="Schwefel", title="2D Schwefel function", figsize=[8, 6],
    levels=15, x_list=None, x_list_d=None, x_list_i=None,
):
    x0 = np.linspace(*x0lims, n)
    x1 = np.linspace(*x1lims, n)
    xx0, xx1 = np.meshgrid(x0, x1)
    X = np.concatenate([xx0.reshape(n, n, 1), xx1.reshape(n, n, 1)], axis=2)

    c = objective(X)
    plt.figure(figsize=figsize)
    plt.contourf(xx0, xx1, c, levels)

    legend = []
    if x_list is not None:
        x0, x1 = zip(*x_list)
        plt.plot(x0, x1, 'ro', alpha=0.3)
        legend.append("Base point")
    if x_list_d is not None:
        x0, x1 = zip(*x_list_d)
        plt.plot(x0, x1, 'go', alpha=1.0, markeredgecolor="k")
        legend.append("Diversification point")
    if x_list_i is not None:
        x0, x1 = zip(*x_list_i)
        plt.plot(x0, x1, 'bo', alpha=0.3, markeredgecolor="k")
        legend.append("Intensification point")

    plt.title(title)
    plt.colorbar()
    plt.axis("tight")
    if len(legend) > 1: plt.legend(legend)
    plt.savefig("Images/" + filename)
    plt.close()

def plot_rosenbrock():
    plot_objective(
        objective=o.rosenbrock, x0lims=[-1.5, 1.5], x1lims=[-1.0, 2.0],
        filename="Images/Rosenbrock", title="Rosenbrock function",
        levels=np.linspace(0, 50, 15)
    )

def plot_fitness_history(
    f_list, filename="Schwefel", title="2D Schwefel function", figsize=[8, 6],
    # f_list_d=None
):
    plt.figure(figsize=figsize)
    plt.plot(f_list, alpha=0.8)

    plt.title(title)
    plt.grid(True)
    # plt.axis("tight")
    plt.xlabel("Evaluation number")
    plt.ylabel("Fitness")
    plt.savefig("Images/" + filename)
    plt.close()

if __name__ == "__main__":
    plot_objective()
    plot_rosenbrock()