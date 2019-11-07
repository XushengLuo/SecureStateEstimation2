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
from itertools import combinations


class TestCase(object):
    def __init__(self, n, p, system, noise):
        self.p = p
        self.n = n
        self.tau = self.n
        # Generate a system with a random A matrix
        self.A = random(self.n, self.n, 0.2)
        # Make sure A has spectral radius 1 (otherwise A^k will be either very large or very small for k large)
        eig, _ = np.linalg.eig(self.A.A)
        self.A = self.A.A / (np.max(np.absolute(eig)) + 0.1)
        # The 'C' matrix of the system
        self.C = random(self.p, self.n, 0.2)
        self.C = self.C.A
        self.Ts = 0.1

        self.sys = ss(self.A, np.zeros((self.n, 1)), self.C, np.zeros((self.p, 1)), self.Ts)
        self.x0 = np.random.randn(self.n, 1)

        # observability matrix
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

        # upper bound for the noise
        self.tol = 1e-4
        if noise == 'noisy':
            self.noise_bound = np.array([1.5]*self.p).reshape(self.p, 1)
        elif noise == 'noiseless':
            self.noise_bound = np.array([0]*self.p).reshape(self.p, 1)

        self.attack_lower_bound = 5

        for percent in [1, 2, 3]:
            self.s = int(self.p * percent / 10)  # np.random.randint(0, self.max_s, 1)[0]

            # Choose a random attacking set K of size qs
            self.per = sorted(range(0, self.p))
            self.K = self.per[0:self.s]

            for r in range(1, 11):
                for scheme in ['random']:
                    self.Y, self.E = self.noisy_attack(noise)
                    with open('data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}worst.mat'.format(self.n, self.p, system, r, scheme, noise, percent),
                              'wb+') as filehandle:
                        pickle.dump(self.Y, filehandle)
                        pickle.dump(self.obsMatrix, filehandle)
                        pickle.dump([self.p, self.n, self.tau], filehandle)
                        pickle.dump(self.K, filehandle)
                        pickle.dump(self.x0, filehandle)
                        pickle.dump(self.E, filehandle)
                        pickle.dump(self.noise_bound, filehandle)
                        pickle.dump(self.A, filehandle)
                        pickle.dump(self.C, filehandle)
                        pickle.dump(self.tol, filehandle)
                        pickle.dump(self.s, filehandle)

    def noisy_attack(self, noise):
        # Choose an initial condition
        x = self.x0
        Y = np.array([]).reshape(self.p, 0)
        E = np.array([]).reshape(self.p, 0)
        if noise == 'noisy':
            noise_power = 0.01
            process_noise_power = 0.05
        elif noise == 'noiseless':
            noise_power = 0
            process_noise_power = 0

        for i in range(0, self.tau):
            # Generate a random attack vector supported on K
            a = np.zeros((self.p, 1))
            a[self.K] = self.attack_lower_bound * np.random.randn(len(self.K), 1)
            E = np.concatenate((E, a), axis=1)

            # The measurement is y=C*x+a
            y = self.C.dot(x) + a + noise_power * np.random.randn(self.p, 1)   # np.random.uniform(-1, 1, (self.p, 1))
            # Update the arrays X,Y,E
            Y = np.concatenate((Y, y), axis=1)

            x = self.A.dot(x) + process_noise_power * np.random.randn(self.n, 1)  # np.random.uniform(-1, 1, (self.n, 1))

        return np.transpose(Y).reshape(np.size(Y), 1, order='F'), np.transpose(E).reshape(np.size(E), 1, order='F')

    def random_attack(self, system, trial, noise):
        for percent in [1, 2, 3]:
            self.s = int(self.p * percent / 10)  # np.random.randint(0, self.max_s, 1)[0]
            per = np.random.permutation(self.p)
            self.K = per[0:self.s]
            for scheme in ['random']:
                self.Y, self.E = self.noisy_attack(noise)
                with open('data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}random.mat'.format(self.n, self.p, system, trial,
                                                                                                 scheme, noise, percent),
                          'wb+') as filehandle:
                    pickle.dump(self.Y, filehandle)
                    pickle.dump(self.obsMatrix, filehandle)
                    pickle.dump([self.p, self.n, self.tau], filehandle)
                    pickle.dump(self.K, filehandle)
                    pickle.dump(self.x0, filehandle)
                    pickle.dump(self.E, filehandle)
                    pickle.dump(self.noise_bound, filehandle)
                    pickle.dump(self.A, filehandle)
                    pickle.dump(self.C, filehandle)
                    pickle.dump(self.tol, filehandle)
                    pickle.dump(self.s, filehandle)


for p in 20 * np.array(range(1, 11)):
    for noise in ['noiseless', 'noisy']:
        for system in range(1, 11):
            print(p, noise, system)
            test_case = TestCase(p, p, system, noise)
            for trial in range(1, 11):
                    test_case.random_attack(system, trial, noise)