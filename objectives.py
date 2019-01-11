import numpy as np

def schwefel(x):
    assert np.all(np.abs(x) <= 500)
    return np.sum(np.prod([x, -np.sin(np.sqrt(np.abs(x)))], axis=0), axis=-1)

def rosenbrock(x):
    assert x.shape[-1] == 2
    return (100 * (x.T[1] - x.T[0] ** 2) ** 2 + (1 - x.T[0]) ** 2).T

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    schwefel(x)
    print(schwefel([1, 2, -403]))
    print(schwefel([[1], [2], [-403]]))