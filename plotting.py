import numpy as np
import matplotlib.pyplot as plt

import objectives as o

def plot_objective(
    objective=o.schwefel, x0lims=[-500, 500], x1lims=[-500, 500], n=200,
    filename="Images/Schwefel", title="2D Schwefel function", figsize=[8, 6],
    levels=15
):
    x0 = np.linspace(*x0lims, n)
    x1 = np.linspace(*x1lims, n)
    xx0, xx1 = np.meshgrid(x0, x1)
    X = np.concatenate([xx0.reshape(n, n, 1), xx1.reshape(n, n, 1)], axis=2)

    c = objective(X)
    plt.figure(figsize=figsize)
    plt.contourf(xx0, xx1, c, levels)
    plt.title(title)
    plt.axis("tight")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def plot_rosenbrock():
    plot_objective(
        objective=o.rosenbrock, x0lims=[-1.5, 1.5], x1lims=[-1.0, 2.0],
        filename="Images/Rosenbrock", title="Rosenbrock function",
        levels=np.linspace(0, 50, 15)
    )

if __name__ == "__main__":
    plot_objective()
    plot_rosenbrock()