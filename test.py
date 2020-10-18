import numpy as np
import sys
np.set_printoptions(threshold=np.inf)
sizes = [2, 4, 3]
sizes2 = [7, 6, 8]
zipped = zip(sizes[:2], sizes[1:])
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
new = zip(biases, weights)
print(biases)
print(weights)
print(list(new))
