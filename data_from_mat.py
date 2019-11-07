"""
__author__ = chrislaw
__project__ = SecureStateEstimation
__date__ = 10/5/18
"""

""" transform data from .mat file to python-used data type
"""
import scipy.io as sio
import numpy as np
import pickle


class TestCase(object):

    def __init__(self, n, p, trial):
        # large scale
        data = sio.loadmat('/Users/chrislaw/Box Sync/Research/SSE/Imhotep-smt-master/ImhotepSMT/Examples/'
                           'Random examples/Test2_sensors/test_n{0}_p{1}_{2}.mat'.format(n, p, trial))

        # data = sio.loadmat('/Users/chrislaw/Box Sync/Research/SSE/Imhotep-smt-master/ImhotepSMT/Examples/'
        #                    'Random examples/Test2_sensors/noise_test_n{0}_p{1}_{2}.mat'.format(n, p, trial))

        self.p = data['p'][0][0]
        self.n = data['n'][0][0]
        self.tau = data['tau'][0][0]
        # Generate a system with a random A matrix
        self.A = data['A']
        # The 'C' matrix of the system
        self.C = data['C']

        self.x0 = data['x0']

        self.attackpower = data['attackpower'][0][0]  # Magnitude of the attacks (i.e., norm of the attack vector)
        # self.max_s = int(np.floor(self.p // 3) - 1)
        self.s = data['s'][0][0]

        self.K = data['K'][0]

        # Choose an initial condition
        Y = np.array([]).reshape(self.p, 0)
        E = np.array([]).reshape(self.p, 0)
        # noise power
        # self.noise_bound = np.array([1.5]*self.p).reshape(self.p, 1) # 2.25

        self.noise_bound = data['noiseBound']
        self.Y = np.transpose(data['Y']).reshape(np.size(data['Y']), 1, order='F')
        self.E = 0
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
# ============================= large scale test without noise =====================================


for n in 20 * np.array(range(10, 11)):
    for trial in range(9, 10):
        print(n)
        testCase = TestCase(n, n, trial)
        with open('data/sse_test_from_mat_n{0}_p{1}_{2}.mat'.format(n, n, trial), 'wb+') as filehandle:
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

# ==================================================================================================
# n = 20
# p = 120
# trial = 1
# testCase = TestCase(n, p, trial)
# with open('data/noise_sse_test_from_mat_n{0}_p{1}_{2}.mat'.format(n, p , trial), 'wb+') as filehandle:
#     pickle.dump(testCase.Y, filehandle)
#     pickle.dump(testCase.obsMatrix, filehandle)
#     pickle.dump([testCase.p, testCase.n, testCase.tau], filehandle)
#     pickle.dump(testCase.K, filehandle)
#     pickle.dump(testCase.x0, filehandle)
#     pickle.dump(testCase.E, filehandle)
#     pickle.dump(testCase.noise_bound, filehandle)
#     pickle.dump(testCase.A, filehandle)
#     pickle.dump(testCase.C, filehandle)
#     pickle.dump(testCase.s, filehandle)
