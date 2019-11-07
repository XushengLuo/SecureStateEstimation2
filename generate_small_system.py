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
    def __init__(self, n, p, trial, noise):
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

        # maximum allowablw number of sensors under attack
        self.max_s = self.maximum_allowable_attacked_sensors()
        if self.max_s == 0:
            return

        self.sys = ss(self.A, np.zeros((self.n, 1)), self.C, np.zeros((self.p, 1)), self.Ts)
        self.x0 = np.random.randn(self.n, 1)

        # noiseless and no attack
        Y_noiseless = self.noiseless()
        # noisy and no attack
        self.noise = np.array([]).reshape(self.p, 0)
        self.process_noise = np.array([]).reshape(self.n, 0)
        Y_noisy = self.noisy_attackless(noise)

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
        # calculate delta_s
        self.delta_s = self.get_delta_s()

        # self.attackpower = 5  # Magnitude of the attacks (i.e., norm of the attack vector)
        # self.max_s = int(np.floor(self.p // 3) - 1)
        self.s = self.max_s  # np.random.randint(0, self.max_s, 1)[0]

        # Choose a random attacking set K of size qs
        self.per = sorted(range(0, self.p))
        self.K = self.per[0:self.s]

        # upper bound for the noise
        self.noise_bound = np.zeros((self.p, 1))
        self.tol = 1e-4
        diff = Y_noisy - Y_noiseless
        for i in range(p):
            self.noise_bound[i, :] = np.linalg.norm(diff[range(i * self.tau, (i + 1) * self.tau), :])
        self.attack_lower_bound = (2 / np.sqrt(1 - self.delta_s)) * np.linalg.norm(self.noise_bound) \
                             + np.sqrt(self.tol) / np.sqrt(1 - self.delta_s)

        for r in range(1, 21):
            for scheme in ['random']:
                self.Y, self.E = self.noisy_attack(scheme)
                with open('smalldata/{6}small_{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_worst.mat'.format(self.n, self.p, trial, r, scheme, noise, def_s),
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
                    pickle.dump(self.max_s, filehandle)
                    pickle.dump(self.delta_s, filehandle)

    def maximum_allowable_attacked_sensors(self):
        max_s = 0
        is_observable = True
        # increase until p//2
        for i in range(self.p//2-1):
            # all combinations
            for j in combinations(range(self.p), 2*(i+1)):
                C_new = np.delete(self.C, j, axis=0)
                if np.linalg.matrix_rank(obsv(self.A, C_new)) != self.n:
                    is_observable = False
                    break
            # if observable
            if is_observable:
                max_s = i + 1
            else:
                break
        return max_s

    def get_delta_s(self):
        max_eig = 0
        # cadinality of index set I
        for i in range(self.p-self.max_s//2, self.p+1):
            # all combinations of index set I
            for index_set_i in combinations(range(self.p), i):
                i_mat = np.zeros((self.n, self.n))
                for ii in index_set_i:
                    obsv_i = self.obsMatrix[range(ii * self.tau, (ii + 1) * self.tau), :]
                    i_mat += np.matmul(np.transpose(obsv_i), obsv_i)
                i_mat_inv = np.linalg.inv(i_mat)
                # cardinality of index set eta
                for j in range(1, self.max_s//2+1):
                    # all combinations of index set eta
                    for index_set_eta in combinations(index_set_i, j):
                        eta_mat = np.zeros((self.n, self.n))
                        for eta in index_set_eta:
                            obsv_eta = self.obsMatrix[range(eta*self.tau, (eta+1)*self.tau), :]
                            eta_mat += np.matmul(np.transpose(obsv_eta), obsv_eta)
                        eigen = np.max(np.linalg.eig(np.matmul(eta_mat, i_mat_inv))[0])
                        if max_eig < eigen:
                            max_eig = eigen
        return max_eig

    def noiseless(self):
        # without noise and attack
        x = self.x0
        Y = np.array([]).reshape(self.p, 0)
        for i in range(0, self.tau):
            # The measurement is y=C*x+a
            y = self.C.dot(x)
            # Update the arrays X,Y,E
            Y = np.concatenate((Y, y), axis=1)

            x = self.A.dot(x)

        Y_noiseless = np.transpose(Y).reshape(np.size(Y), 1, order='F')
        return Y_noiseless

    def noisy_attackless(self, noi):
        # noisby but no attack
        # Choose an initial condition
        x = self.x0
        Y = np.array([]).reshape(self.p, 0)
        noise = np.array([]).reshape(self.p, 0)
        process_noise = np.array([]).reshape(self.n, 0)
        # noise power
        if noi == 'noisy':
            noise_power = 0.01
            process_noise_power = 0.05
        elif noi == 'noiseless':
            noise_power = 0.00
            process_noise_power = 0.00

        for i in range(0, self.tau):
            process_noise_t = 0 * np.random.randn(self.n, 1)
            process_noise = np.concatenate((process_noise, process_noise_t), axis=1)

            # The measurement is y=C*x+a
            noise_t = noise_power * np.random.randn(self.p, 1)
            noise = np.concatenate((noise, noise_t), axis=1)
            y = self.C.dot(x) + noise_t
            # Update the arrays X,Y,E
            Y = np.concatenate((Y, y), axis=1)

            process_noise_t = process_noise_power * np.random.randn(self.n, 1)
            x = self.A.dot(x) + process_noise_t

        self.noise = noise
        self.process_noise = process_noise

        Y_noisy = np.transpose(Y).reshape(np.size(Y), 1, order='F')
        return Y_noisy

    def noisy_attack(self, scheme):
        # Choose an initial condition
        x = self.x0
        Y = np.array([]).reshape(self.p, 0)

        # attack signal matrix
        E = np.zeros((self.p, self.tau))
        for k in self.K:
            # stealty
            if scheme == 'stealthy':
                a = np.random.randn(1, self.tau)
                a = self.noise_bound[np.random.randint(self.p, size=1), :] * a / np.linalg.norm(a)
            # random
            elif scheme == 'random':
                a = np.random.randn(1, self.tau)
                a = self.attack_lower_bound * a / np.linalg.norm(a)
            # bias
            elif scheme == "constant":
                a = self.attack_lower_bound / np.sqrt(self.tau)
            E[k, :] = a

        for i in range(0, self.tau):
            # The measurement is y=C*x+a
            y = self.C.dot(x) + np.reshape(E[:, i] + self.noise[:, i], (self.p, 1))
            # Update the arrays X,Y,E
            Y = np.concatenate((Y, y), axis=1)
            x = self.A.dot(x) + np.reshape(self.process_noise[:, i], (self.n, 1))

        return np.transpose(Y).reshape(np.size(Y), 1, order='F'), np.transpose(E).reshape(np.size(E), 1, order='F')
        # Y = [ Y_1^t
        #       Y_2^t           Y_i^t = []_(self.tau, 1)
        #        ...
        #       Y_p^t ]

    def random_attack(self, trial, r, noise, def_s):
        per = np.random.permutation(self.p)
        self.K = per[0:self.s]
        for scheme in ['random']:
            self.Y, self.E = self.noisy_attack(scheme)
            with open('smalldata/{6}small_{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_random.mat'.format(self.n, self.p, trial, r, scheme, noise, def_s),
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
                pickle.dump(self.max_s, filehandle)
                pickle.dump(self.delta_s, filehandle)


for noise in ['noiseless', 'noisy']:
    for def_s in [2, 3, 4]:
        trial = 1
        n = trial + 50
        while trial < n:
            test_case = TestCase(10, 10, trial, noise)
            if test_case.max_s != def_s:
                continue
            try:
                print(noise, trial, test_case.max_s, test_case.delta_s, 2/np.sqrt(1-test_case.delta_s))
                if 2/np.sqrt(1-test_case.delta_s) > 1000:
                    continue
                for r in range(1, 21):
                    test_case.random_attack(trial, r, noise, def_s)
                trial += 1
            except AttributeError:
                pass
