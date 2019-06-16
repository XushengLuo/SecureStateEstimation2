import numpy as np
from generate_test_case import TestCase
import itertools
import pickle


class SecureStateEsitmation:
    def __init__(self):
        self.tol = 1e-5
        # performance of tolorance, with larger tol, it is susceptible to errors,
        # treating attacked as attacked-free and attacked-free as attacked.
        # with open('sse_test_from_mat', 'rb') as filehandle:
        with open('sse_test', 'rb') as filehandle:
            self.Y = pickle.load(filehandle)
            self.obsMatrix = pickle.load(filehandle)
            self.p, self.n, self.tau = pickle.load(filehandle)
            self.K = pickle.load(filehandle)
            self.x0 = pickle.load(filehandle)
            self.E = pickle.load(filehandle)
            self.noise_bound = pickle.load(filehandle)
            self.A = pickle.load(filehandle)
            self.C = pickle.load(filehandle)
            self.s = pickle.load(filehandle)

# ------------------monitor the power of noise, \bar{w}---------------------
# for i in range(100):
#     w = - np.inf
#     testCase = TestCase()
#     with open('sse_test', 'wb') as filehandle:
#         pickle.dump(testCase.Y, filehandle)
#         pickle.dump(testCase.obsMatrix, filehandle)
#         pickle.dump([testCase.p, testCase.n, testCase.tau], filehandle)
#         pickle.dump(testCase.K, filehandle)
#         pickle.dump(testCase.x0, filehandle)
#         pickle.dump(testCase.E, filehandle)
#         pickle.dump(testCase.noise_bound, filehandle)
#         pickle.dump(testCase.A, filehandle)
#         pickle.dump(testCase.C, filehandle)
#         pickle.dump(testCase.s, filehandle)
#
#     sse = SecureStateEsitmation()
#     for j in range(sse.p):
#         index = [x + y for x, y in itertools.product([j * sse.tau], range(sse.tau))]
#         Y = sse.Y[index, :]
#         O = sse.obsMatrix[index, :]
#         e = np.linalg.norm(Y - O.dot(sse.x0))
#     # print(e)
#     if e > w:
#         w = e
#     print(w)


for i in range(100):
    testCase = TestCase()
    with open('sse_test', 'wb') as filehandle:
        pickle.dump(testCase.Y, filehandle)
        pickle.dump(testCase.obsMatrix, filehandle)
        pickle.dump([testCase.p, testCase.n, testCase.tau], filehandle)
        pickle.dump(testCase.K, filehandle)
        pickle.dump(testCase.x0, filehandle)
        pickle.dump(testCase.E, filehandle)
        pickle.dump(testCase.noise_bound, filehandle)
        pickle.dump(testCase.A, filehandle)
        pickle.dump(testCase.C, filehandle)
        pickle.dump(testCase.s, filehandle)
    sse = SecureStateEsitmation()
    Delta = -1 * np.inf
    # Choose a random attacking set K of size qs
    for i in range(sse.p - sse.s, sse.p + 1):  # (p-s, p) for matrix I
        # print(i)
        l = list(itertools.combinations(range(sse.p), i))
        # print(len(l))
        for ls in l:
            index = [x + y for x, y in itertools.product([k * sse.tau for k in ls], range(sse.tau))]
            O2 = sse.obsMatrix[index, :]
            S2 = np.transpose(O2).dot(O2)
            for j in range(sse.s + 1):  # (0, s) for matrix gamma
                ll = list(itertools.combinations(ls, j))

                for lsl in ll:
                    index1 = [x + y for x, y in itertools.product([k * sse.tau for k in lsl], range(sse.tau))]
                    O1 = sse.obsMatrix[index1, :]
                    S1 = np.transpose(O1).dot(O1)
                    try:
                        eig, _ = np.linalg.eig(S1.dot(np.linalg.inv(S2)))
                        e = np.max(eig).real
                        # print(e)
                        if e > Delta:
                            Delta = e
                    except np.linalg.linalg.LinAlgError:
                        continue

    de = np.sqrt(1 - Delta)
    print(de)
    # print(2 / de * 1.5 + np.sqrt(1e-5) / de)

# ----------------check the maximum eigenvalue of the matrix ----------------------
# sse = SecureStateEsitmation()
#
# # Choose a random attacking set K of size qs
# per = np.random.permutation(sse.s)
#
# index = [x + y for x, y in itertools.product([i * sse.tau for i in per], range(sse.tau))]
# O1 = sse.obsMatrix[index, :]
#
# S1 = np.transpose(O1).dot(O1)
# S2 = np.transpose(sse.obsMatrix).dot(sse.obsMatrix)
# eig, _ = np.linalg.eig(S1.dot(np.linalg.inv(S2)))
# e = np.max(eig).real
# print(e)

# -----------fix the system matrices then change time window or reorder-----------------
#
# from scipy.sparse import random
# import numpy as np
# from control.matlab import *
# import pickle
#
#
# class TestCase(object):
#     def __init__(self):
#         with open('sse_test_worst', 'rb') as filehandle:
#         # # with open('sse_test', 'rb') as filehandle:
#                     self.Y = pickle.load(filehandle)
#                     self.obsMatrix = pickle.load(filehandle)
#                     self.p, self.n, self.tau = pickle.load(filehandle)
#                     self.tau = self.tau//3
#                     self.K = pickle.load(filehandle)
#                     self.x0 = pickle.load(filehandle)
#                     self.E = pickle.load(filehandle)
#                     self.noise_bound = pickle.load(filehandle)
#                     self.A = pickle.load(filehandle)
#                     self.C = pickle.load(filehandle)
#                     self.s = pickle.load(filehandle)
#
#         # Generate a system with a random A matrix
#
#         self.Ts = 0.1
#
#         self.sys = ss(self.A, np.zeros((self.n, 1)), self.C, np.zeros((self.p, 1)), self.Ts)
#         self.x0 = np.random.randn(self.n, 1)
#
#         self.attackpower = 20 # Magnitude of the attacks (i.e., norm of the attack vector)
#
#         # Choose a random attacking set K of size qs
#         self.per = np.random.permutation(self.p)
#         self.K = self.per[0:self.s]
#         # print(sorted(self.K))
#
#         # Choose an initial condition
#         x = self.x0
#         Y = np.array([]).reshape(self.p, 0)
#         E = np.array([]).reshape(self.p, 0)
#         # noise power
#         noise_power = 0 #.01
#         process_noise_power = 0 # .05
#         self.noise_bound = np.array([0]*self.p).reshape(self.p, 1) # when noise = 0.01 process noise = 0.05, w_i = 3
#
#         for i in range(0, self.tau):
#             # Generate a random attack vector supported on K
#             a = np.zeros((self.p, 1))
#             a[self.K] = self.attackpower * np.random.randn(len(self.K), 1)
#             E = np.concatenate((E, a), axis=1)
#
#             # The measurement is y=C*x+a
#             y = self.C.dot(x) + a + noise_power * np.random.randn(self.p, 1)  # np.random.uniform(-1, 1, (self.p, 1))
#             # Update the arrays X,Y,E
#             Y = np.concatenate((Y, y), axis=1)
#
#             x = self.A.dot(x) + process_noise_power * np.random.randn(self.n, 1) # np.random.uniform(-1, 1, (self.n, 1))
#
#         self.Y = np.transpose(Y).reshape(np.size(Y), 1, order='F')
#         self.E = np.transpose(E).reshape(np.size(E), 1, order='F')
#         # Y = [ Y_1^t
#         #       Y_2^t           Y_i^t = []_(t, 1)
#         #        ...
#         #       Y_p^t ]
#
#         self.obsMatrix = np.array([]).reshape(0, self.n)
#
#         for k in range(self.p):
#             obs = self.C[k, :].reshape(1, self.n)
#             oi = np.array([]).reshape(0, self.n)
#             for i in range(0, self.tau):
#                 obs = obs.dot(self.A) if i else obs
#                 oi = np.concatenate((oi, obs), axis=0)
#
#             self.obsMatrix = np.concatenate((self.obsMatrix, oi), axis=0)
#             # O = [ O_1
#             #       O_2        O_i = []_(self.tau, self.n)
#             #       ...
#             #       O_p ]


# lazy search
# depth first

# testCase = TestCase()
# with open('sse_test', 'wb') as filehandle:
#             pickle.dump(testCase.Y, filehandle)
#             pickle.dump(testCase.obsMatrix, filehandle)
#             pickle.dump([testCase.p, testCase.n, testCase.tau], filehandle)
#             pickle.dump(testCase.K, filehandle)
#             pickle.dump(testCase.x0, filehandle)
#             pickle.dump(testCase.E, filehandle)
#             pickle.dump(testCase.noise_bound, filehandle)
#             pickle.dump(testCase.A, filehandle)
#             pickle.dump(testCase.C, filehandle)
#             pickle.dump(testCase.s, filehandle)


