#!/usr/bin/python

# Copyright 2018, Gurobi Optimization, LLC
"""
Use Gurobi to solve the MIQCP
"""
from gurobipy import *
import pickle
import numpy as np
import scipy.linalg as la
from itertools import product
import datetime
import sys
from generate_test_case import TestCase


class SecureStateEsitmation:
    def __init__(self, n, p, system, r, noise, percent, selection):
        # performance of tolorance, with larger tol, it is susceptible to errors,
        # treating attacked as attacked-free and attacked-free as attacked.
        with open('data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}{7}.mat'.format(n, p, system, r, 'random', noise,
                                                                                       percent, selection),
                  'rb+') as filehandle:
            self.Y = pickle.load(filehandle)
            self.obsMatrix = pickle.load(filehandle)
            self.p, self.n, self.tau = pickle.load(filehandle)
            self.K = pickle.load(filehandle)
            self.x0 = pickle.load(filehandle)
            self.E = pickle.load(filehandle)
            self.noise_bound = pickle.load(filehandle)
            self.A = pickle.load(filehandle)
            self.C = pickle.load(filehandle)
            self.tol = pickle.load(filehandle)
            # with open('sse_test', 'rb') as filehandle:
        #     self.Y = pickle.load(filehandle)
        #     self.obsMatrix = pickle.load(filehandle)
        #     self.p, self.n, self.tau = pickle.load(filehandle)
        #     self.K = pickle.load(filehandle)
        #     self.x0 = pickle.load(filehandle)
        #     self.E = pickle.load(filehandle)
        #     self.noise_bound = pickle.load(filehandle)
        #     self.A = pickle.load(filehandle)
        #     self.C = pickle.load(filehandle)


noise_ = ['noisy', 'noiseless']
selection_ = ['random', 'worst']
p = int(sys.argv[1])
noise = noise_[int(sys.argv[2])]
selection = selection_[int(sys.argv[3])]
percent = int(sys.argv[4])

# for noise in ['noisy', 'noiseless']:
#     for selection in ['random', 'worst']:
#         p = int(sys.argv[1])
#         for percent in range(1, 5):
mistake = []
time = []
error = []
for system in range(1, 6):
    for r in range(1, 6):
        # randomly generate data
        # testCase = TestCase()
        # with open('sse_test', 'wb') as filehandle:
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
        start = datetime.datetime.now()
        sse = SecureStateEsitmation(p, p, system, r, noise, percent, selection)

        # Create a new model
        m = Model("qp")

        M = 1e8

        # Create variables
        b = [m.addVar(vtype=GRB.BINARY, name='b{0}'.format(i + 1)) for i in range(sse.p)]
        b = np.array(b)
        x = [m.addVar(name='x{0}'.format(i + 1)) for i in range(sse.n)]
        x = np.array(x)
        # set object
        obj = LinExpr()
        for i in range(sse.p):
            obj = obj + b[i]
        m.setObjective(obj, GRB.MINIMIZE)
        m.update()
        # constraints
        for i in range(sse.p):
            index = [i * sse.tau + j for j in range(sse.tau)]
            Y = sse.Y[index, :]
            O = sse.obsMatrix[index, :]
            # take the square of l2 norm
            m.addQConstr(
                (-2 * np.transpose(Y).dot(O).dot(x) - M * b[i] + np.transpose(x).dot(np.transpose(O).dot(O).dot(x)))[
                    0] <=
                -1 * np.transpose(Y).dot(Y)[0][0] + 1e-5 + 3**2, name='c{0}'.format(i))  # 1.5**2

        m.setParam('OutputFlag', False)
        # m.Params.Aggregate = 0
        # m.Params.numericFocus = 3
        # m.Params.presolve = 0
        m.Params.MIPGap = 1e-6
        m.optimize()
        time.append((datetime.datetime.now() - start).total_seconds())
        estimate = []
        indexOfZero = []
        indexOfOne = []
        id = 0
        for v in m.getVars():
            if 'b' in v.varName and v.x == 0:
                indexOfZero.append(id)
            if v.x == 1:
                indexOfOne.append(id)
            if 'x' in v.varName:
                estimate.append(v.x)
            id = id + 1

        # print(indexOfOne)
        # print(sorted(sse.K))
        print("true attack ({0})        : ".format(len(sse.K)), sorted(sse.K))
        print("estimate attack ({0})    : ".format(len(indexOfOne)), [i for i in indexOfOne])
        if sorted(sse.K) != [i for i in indexOfOne]:
            mistake.append((noise, selection, percent, system, r))
            continue
        index = [x + y for x, y in product([1 * i * sse.tau for i in indexOfZero], range(sse.tau))]
        Y = sse.Y[index, :]
        O = sse.obsMatrix[index, :]
        x, res, _, _ = la.lstsq(O, Y)
        error.append(np.linalg.norm(x - sse.x0) / np.linalg.norm(sse.x0))

        print('MIQP: {3}, {4}, {5}, n: {0}, error: {1}, time: {2}'.format(p, np.mean(error), np.mean(time), noise, selection, percent))
with open('result/MIQP_{0}_{1}_{2}_{3}.mat'.format(noise, selection, p, percent), 'wb+') as filehandle:
    pickle.dump(time, filehandle)
    pickle.dump(error, filehandle)
    pickle.dump(mistake, filehandle)



# An useful example of the use of Gorobi can be found at
# https://github.com/kehlert/multistage_portfolio/blob/d5c9c040e095330fd68efda050803572a41093ae/code/markowitz.py
