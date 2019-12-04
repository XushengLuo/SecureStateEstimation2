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
from scipy.io import savemat
import sys


class TestCase(object):
    def __init__(self, n, p, system, noise):
        self.p = p
        self.n = n
        self.tau = self.n
        # Generate a system with a random A matrix
        self.A = random(self.n, self.n, 0.3)
        # Make sure A has spectral radius 1 (otherwise A^k will be either very large or very small for k large)
        eig, _ = np.linalg.eig(self.A.A)
        self.A = self.A.A / (np.max(np.absolute(eig)) + 0.1)
        # The 'C' matrix of the system
        self.C = random(self.p, self.n, 0.3)
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
        nb = {20: 0.6797814767648725,
                40: 1.8461709451533266,
                60: 2.9679462841315716,
                80: 3.920406331162999,
                100: 5.076222281880476,
                120: 6.654829334584581,
                140: 9.744176332809849,
                160: 11.300367969361854,
                180: 10.89508884181865,
                200: 18.625466747633144}
        if noise == 'noisy':
            self.noise_bound = np.array([nb[p]] * self.p).reshape(self.p, 1)
        elif noise == 'noiseless':
            self.noise_bound = np.array([0] * self.p).reshape(self.p, 1)

        self.attack_lower_bound = 20

        for percent in range(1, 4):
            self.s = int(self.p * percent / 10)  # np.random.randint(0, self.max_s, 1)[0]

            # Choose a random attacking set K of size qs
            self.per = sorted(range(0, self.p))
            self.K = self.per[0:self.s]

            for r in range(1, 6):
                for scheme in ['random']:
                    self.Y, self.E = self.noisy_attack(noise)
                    with open('data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}worst.mat'.format(self.n, self.p,
                                                                                                     system, r, scheme,
                                                                                                     noise, percent),
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

                    savemat(
                        '/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/'
                        'Random examples/Test2_sensors/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}worst.mat'.format(self.n,
                                                                                                          self.p,
                                                                                                          system, r,
                                                                                                          scheme, noise,
                                                                                                          percent),
                        mdict={'Y': self.Y, 'x0': self.x0, 'K': self.K,
                               'A': self.A, 'C': self.C, 'noiseBound': self.noise_bound, 's': self.s})

    def noisy_attack(self, noise):
        # Choose an initial condition
        x = self.x0
        Y = np.array([]).reshape(self.p, 0)
        E = np.array([]).reshape(self.p, 0)
        if noise == 'noisy':
            noise_power = 0.01
            process_noise_power = 0.01
        elif noise == 'noiseless':
            noise_power = 0
            process_noise_power = 0

        for i in range(0, self.tau):
            # Generate a random attack vector supported on K
            a = np.zeros((self.p, 1))
            a[self.K] = self.attack_lower_bound * np.random.randn(len(self.K), 1)
            E = np.concatenate((E, a), axis=1)

            # The measurement is y=C*x+a
            y = self.C.dot(x) + a + noise_power * np.random.randn(self.p, 1)  # np.random.uniform(-1, 1, (self.p, 1))
            # Update the arrays X,Y,E
            Y = np.concatenate((Y, y), axis=1)

            x = self.A.dot(x) + process_noise_power * np.random.randn(self.n,
                                                                      1)  # np.random.uniform(-1, 1, (self.n, 1))

        return np.transpose(Y).reshape(np.size(Y), 1, order='F'), np.transpose(E).reshape(np.size(E), 1, order='F')

    def random_attack(self, system, trial, noise):
        for percent in range(1, 4):
            self.s = int(self.p * percent / 10)  # np.random.randint(0, self.max_s, 1)[0]
            per = np.random.permutation(self.p)
            self.K = per[0:self.s]
            for scheme in ['random']:
                self.Y, self.E = self.noisy_attack(noise)
                with open(
                        'data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}random.mat'.format(self.n, self.p, system,
                                                                                                trial,
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

                savemat('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/'
                        'Random examples/Test2_sensors/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}random.mat'.format(
                    self.n, self.p, system, trial,
                    scheme, noise, percent),
                    mdict={'Y': self.Y, 'x0': self.x0, 'K': self.K,
                           'A': self.A, 'C': self.C, 'noiseBound': self.noise_bound, 's': self.s})


p = int(sys.argv[1])
for noise in ['noisy', 'noiseless']:
    for system in range(1, 6):
        print(p, noise, system)
        test_case = TestCase(p, p, system, noise)
        for trial in range(1, 6):
            test_case.random_attack(system, trial, noise)


# 20 0.6797814767648725
# 40 1.8461709451533266
# 60 2.9679462841315716
# 80 3.920406331162999
# 100 5.076222281880476
# 120 6.654829334584581
# 140 9.744176332809849
# 160 11.300367969361854
# 180 10.89508884181865
# 200 18.625466747633144