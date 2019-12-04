import pickle
from itertools import product
import numpy as np
import scipy.linalg as la
import sys

noise = 'noisy'
p = int(sys.argv[1])
m = 0
n = p
for selection in ['random', 'worst']:
    for percent in [1, 2, 3]:
        for system in range(1, 11):
            for r in range(1, 11):
                with open('data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}{7}.mat'.format(n, p, system, r,
                                                                                               'random', noise,
                                                                                               percent,
                                                                                               selection),
                          'rb+') as filehandle:
                    Y = pickle.load(filehandle)
                    obsMatrix = pickle.load(filehandle)
                    p, n, tau = pickle.load(filehandle)
                    K = pickle.load(filehandle)
                    x0 = pickle.load(filehandle)
                    E = pickle.load(filehandle)
                    noise_bound = pickle.load(filehandle)
                    A = pickle.load(filehandle)
                    C = pickle.load(filehandle)
                    tol = pickle.load(filehandle)
                for k in range(p):
                    if k not in K:
                        index = [x + y for x, y in product([k*tau], range(tau))]
                        Y_ = Y[index, :]
                        O = obsMatrix[index, :]
                        nb = np.linalg.norm(Y_-O.dot(x0))
                        if m < nb:
                            m = nb
print(p, m)