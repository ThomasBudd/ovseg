import numpy as np

def f(l):
    return np.sum(np.array(l) * np.array([4, 8, 16]))

print(f([6, 3, 2]))

print(f([4, 4, 2]))

print(f([2, 3, 3]))
