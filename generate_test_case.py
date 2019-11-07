"""
__author__ = chrislaw
__project__ = SecureStateEstimation
__date__ = 9/18/18
"""
""" generate data for future use
"""

from scipy.sparse import random
import numpy as np
from control.matlab import *
import pickle


class TestCase(object):
    def __init__(self, n, p):
        self.p = p
        self.n = n
        self.tau = self.n
        # Generate a system with a random A matrix
        self.A = random(self.n, self.n, 0.3)
        # Make sure A has spectral radius 1 (otherwise A^k will be either very large or very small for k large)
        eig, _ = np.linalg.eig(self.A.A)
        self.A = self.A.A / (np.max(np.absolute(eig)) + 0.1)
        # The 'C' matrix of the system
        self.C = random(self.p, self.n, 0.2)
        self.C = self.C.A
        self.Ts = 0.1

        self.sys = ss(self.A, np.zeros((self.n, 1)), self.C, np.zeros((self.p, 1)), self.Ts)
        self.x0 = np.random.randn(self.n, 1)

        self.attackpower = 5 # Magnitude of the attacks (i.e., norm of the attack vector)
        self.max_s = int(np.floor(self.p // 3) - 1)
        self.s = self.max_s # np.random.randint(0, self.max_s, 1)[0]

        # Choose a random attacking set K of size qs
        self.per = np.random.permutation(self.p)
        # self.per = sorted(range(0, self.p))
        self.K = self.per[0:self.s]

        # Choose an initial condition
        x = self.x0
        Y = np.array([]).reshape(self.p, 0)
        E = np.array([]).reshape(self.p, 0)
        # noise power
        noise_power = 0 # 0.01
        process_noise_power = 0 # 0.05
        self.noise_bound = np.array([0]*self.p).reshape(self.p, 1) # when noise = 0.01 process noise = 0.05, w_i = 1.5

        for i in range(0, self.tau):
            # Generate a random attack vector supported on K
            a = np.zeros((self.p, 1))
            a[self.K] = self.attackpower * np.random.randn(len(self.K), 1)
            E = np.concatenate((E, a), axis=1)

            # The measurement is y=C*x+a
            y = self.C.dot(x) + a + noise_power * np.random.randn(self.p, 1)  # np.random.uniform(-1, 1, (self.p, 1))
            # Update the arrays X,Y,E
            Y = np.concatenate((Y, y), axis=1)

            x = self.A.dot(x) + process_noise_power * np.random.randn(self.n, 1) # np.random.uniform(-1, 1, (self.n, 1))

        self.Y = np.transpose(Y).reshape(np.size(Y), 1, order='F')
        self.E = np.transpose(E).reshape(np.size(E), 1, order='F')
        # Y = [ Y_1^t
        #       Y_2^t           Y_i^t = []_(t, 1)
        #        ...
        #       Y_p^t ]

        self.obsMatrix = np.array([]).reshape(0, self.n)

        for k in range(self.p):
            obs = self.C[k, :].reshape(1, self.n)
            oi = np.array([]).reshape(0, self.n)
            for i in range(0, self.tau):
                obs = obs.dot(self.A) if i else obs
                oi = np.concatenate((oi, obs), axis=0)

            self.obsMatrix = np.concatenate((self.obsMatrix, oi), axis=0)
            # O = [ O_1
            #       O_2        O_i = []_(self.tau, self.n)
            #       ...
            #       O_p ]


for p in 20 * np.array(range(1, 11)):
    for trial in range(1, 11):
        testCase = TestCase(50, p)
        with open('data/sse_test_from_mat_n{0}_p{1}_{2}.mat'.format(testCase.n, testCase.p, trial),
                  'wb+') as filehandle:
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


