"""
__author__ = chrislaw
__project__ = SecureStateEstimation
__date__ = 9/26/18
"""
"""
 test complexity for test case I: complexity and optimality
"""
import queue
import datetime
import numpy as np
import pickle
from itertools import product
import scipy.linalg as la
import warnings
import pickle
import sys


class SecureStateEsitmation:
    def __init__(self, n, p, system, r, noise, percent, selection):
        # performance of tolorance, with larger tol, it is susceptible to errors,
        # treating attacked as attacked-free and attacked-free as attacked.
        with open('data/{5}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{6}{7}.mat'.format(n, p, system, r, 'random', noise, percent, selection),
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

    def obs(self):
        # whether the system is observable
        import control
        obs = control.obsv(self.A, self.C)
        rank = np.linalg.matrix_rank(obs)
        print("rank", rank, "n", self.n)

    def residual(self, indexOfZero):
        index = [x + y for x, y in product([-1 * i * self.tau for i in indexOfZero], range(self.tau))]
        Y = self.Y[index, :]
        O = self.obsMatrix[index, :]
        x, res, _, _ = la.lstsq(O, Y)
        res = np.linalg.norm(Y - O.dot(x))
        # attackfree = [-1 * i + 1 for i in indexOfZero]

        # attackfree = [-1 * i + 1 for i in indexOfZero]
        # attack = [i - 1 for i in range(1, self.p + 1) if i not in attackfree]
        # print(attack)
        # if attack == [6, 12, 20, 32, 39, 42, 67, 70, 72, 73, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
        #               118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
        #               138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]:

            # print(res, self.tol + np.linalg.norm(self.noise_bound[np.array(indexOfZero) * -1, :]))
        if res <= self.tol + np.linalg.norm(self.noise_bound[np.array(indexOfZero) * -1, :]):
            return True
        else:  # no intersection point
            return False

    def genChild(self, parentnode, childnode, attack):
        """Generating childnote
        """
        childnode.attack = attack
        childnode.level = parentnode.level - 1  # negative, convenient when enqueued
        childnode.parent = parentnode
        childnode.numOfAttacked = parentnode.numOfAttacked + attack
        childnode.indexOfZero = parentnode.indexOfZero + [childnode.level] if not attack else parentnode.indexOfZero
        childnode.accmuResidual = True if attack else self.residual(childnode.indexOfZero)


class Node:
    """Including state, parent, pathcost"""

    def __init__(self, acr=True, noa=0, level=0, attack=1, ioo=list(), par=None):
        self.numOfAttacked = noa
        self.level = level
        self.accmuResidual = acr
        self.attack = attack
        self.indexOfZero = ioo
        self.parent = par

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.attack == other.attack and self.level == other.level
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, Node):
            if self.numOfAttacked < other.numOfAttacked or (
                    self.numOfAttacked == other.numOfAttacked and self.level < other.level):
                return True
            else:
                return False

    def __hash__(self):
        return hash((self.attack, self.level))


def main(n, p, system, r, noise, percent, selection):

    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    level = []
    # Request the init and goal state
    sse = SecureStateEsitmation(n, p, system, r, noise, percent, selection)

    # Initializing root node
    root = Node(acr=True, noa=0, level=1, attack=0, ioo=[], par=None)

    # Initializing frontier
    frontier = queue.PriorityQueue()
    discard = queue.PriorityQueue()   # for lazy research when previous search fails
    # a priority queue ordered, with node as the only element
    frontier.put(root)
    # Initializing explored set
    exploredSet = set()  # set
    # iteration
    itera = 0
    while True:

        if (datetime.datetime.now() - start).total_seconds() > 200:
            return False, False

        itera = itera + 1
        # EMPTY?(frontier) and EMPTY?(discard), then return failure
        if frontier.empty() and discard.empty():
            break
        if frontier.empty():
            frontier.put(discard.get())
            exploredSet.clear()
            continue

            # break

        # chooses the lowest-cost node in frontier
        node = frontier.get()
        # print(node.level * -1 + 1, node.attack, [i * -1 + 1 for i in node.indexOfZero])
        level.append(node.level * -1 + 1)

        # stop condition: when there is no state cost in the frontier
        # less than the temporary optimal cost

        if node.level == -1 * (sse.p - 1):
            index = [x + y for x, y in product([-1 * i * sse.tau for i in node.indexOfZero], range(sse.tau))]
            Y = sse.Y[index, :]
            O = sse.obsMatrix[index, :]
            x, res, _, _ = la.lstsq(O, Y)
            # print("error: ", np.linalg.norm(x - sse.x0)/np.linalg.norm(sse.x0))
            attackfree = [-1 * i + 1 for i in node.indexOfZero]
            attack = [i-1 for i in range(1, sse.p + 1) if i not in attackfree]

            # sse_from_mat i otherwise i+1
            print("true attack ({0})        : ".format(len(sse.K)), sorted([i for i in sse.K]))
            print("estimate attack ({0})    : ".format(len(attack)), attack)
            time = (datetime.datetime.now() - start).total_seconds()
            # wrong attack detection
            if attack != sorted([i for i in sse.K]):
                print("true attack ({0})        : ".format(len(sse.K)), sorted([i for i in sse.K]))
                print("estimate attack ({0})    : ".format(len(attack)), attack)
                return False, False
                # break
            # ----------------- date from matlab ---------------
            # print("true attack ({0})        : ".format(len(sse.K)), sorted([i+1 for i in sse.K]))
            # print("estimate attack ({0})    : ".format(len(attack)), attack)
            # --------------------------------------------------
            return np.linalg.norm(x - sse.x0)/np.linalg.norm(sse.x0), time, itera

        exploredSet.add(node)

        for attack in [0, 1]:

            # child CHILD-NODE(problem,node,action)
            childNode = Node()
            sse.genChild(node, childNode, attack)

            if childNode.accmuResidual:

                # # early stop
                if (-1 * childNode.level + 1) - len(childNode.indexOfZero) > sse.p // 2 - 1: # we only discard bad nodes
                    continue

                # if childNode in exploredSet:
                #     discard.put(childNode)
                #     continue
                # # only consider 0 residual
                #
                # # option 2
                # q = frontier.queue
                # if childNode not in q:
                #     # print("childnode accepted: ", [i * -1 + 1 for i in childNode.indexOfZero], childNode.level * -1 + 1)
                #     frontier.put(childNode)
                # else:
                #     discard.put(childNode)

                if childNode in exploredSet or childNode in frontier.queue:
                    discard.put(childNode)
                    continue
                else:
                    frontier.put(childNode)
    return False, False


if __name__ == "__main__":
    # -------------- level v.s. iteration ---------------
    # from Delta import TestCase
    # level = []
    # for i in range(1000):
    #     # print(i)
    #     testCase = TestCase()
    #     with open('sse_test_worst1', 'wb') as filehandle:
    #         pickle.dump(testCase.Y, filehandle)
    #         pickle.dump(testCase.obsMatrix, filehandle)
    #         pickle.dump([testCase.p, testCase.n, testCase.tau], filehandle)
    #         pickle.dump(testCase.K, filehandle)
    #         pickle.dump(testCase.x0, filehandle)
    #         pickle.dump(testCase.E, filehandle)
    #         pickle.dump(testCase.noise_bound, filehandle)
    #         pickle.dump(testCase.A, filehandle)
    #         pickle.dump(testCase.C, filehandle)
    #     start = datetime.datetime.now()
    #     l = main()
    #     level.append(l)
    #     print(l)
    # print(np.mean(level), np.min(level), np.max(level))

    # ------------- multiple time runtime --------------
    # ===================== large scale test ================================
    # from generate_test_case import TestCase

    # for noise in ['noisy', 'noiseless']:
    #     for selection in ['random', 'worst']:
    noise_ = ['noisy', 'noiseless']
    selection_ = ['random', 'worst']
    p = int(sys.argv[1])
    noise = noise_[int(sys.argv[2])]
    selection = selection_[int(sys.argv[3])]
    percent = int(sys.argv[4])
    n = int(sys.argv[5])
    time = []
    error = []
    itera = []
    mistake = []
    # noise = 'noisy'
    # selection = 'random'
    # p = 200
    # percent = 1
    for system in range(1, 6):
        for r in range(1, 6):
            start = datetime.datetime.now()
            # e, t, i = main(p, p, system, r, noise, percent, selection)
            try:
                e, t, i = main(n, p, system, r, noise, percent, selection)
            except ValueError:
                mistake.append((noise, selection, percent, system, r))
                continue
            if not t:
                print('=== Something goes wrong.. maybe noise too large or attack too weak ==')
                continue
            time.append(t)
            error.append(e)
            itera.append(i)
            print('SSE:', noise, selection, p, percent, e, t, i, system, r)

    with open('result/complexity_{0}_{1}_{2}_{3}_{4}.mat'.format(noise, selection, p, n, percent), 'wb+') as filehandle:
        pickle.dump(time, filehandle)
        pickle.dump(error, filehandle)
        pickle.dump(itera, filehandle)
        pickle.dump(mistake, filehandle)
    # ----------------- date from matlab ---------------
    # from data_from_mat import TestCase
    # testCase = TestCase()
    # time = []
    # for i in range(1):
    #     start = datetime.datetime.now()
    #     t, e = main()
    #     time.append(t)
    # print(e, np.mean(time))
    # -----------------------------------------------------

    # from generate_test_case import TestCase
    # n = 20
    # p = 120
    # trial = 1
    # time = []
    # error = []
    # for trial in range(1, 2):
    #     # testCase = TestCase()
    #     # with open('sse_test', 'wb') as filehandle:
    #     #     pickle.dump(testCase.Y, filehandle)
    #     #     pickle.dump(testCase.obsMatrix, filehandle)
    #     #     pickle.dump([testCase.p, testCase.n, testCase.tau], filehandle)
    #     #     pickle.dump(testCase.K, filehandle)
    #     #     pickle.dump(testCase.x0, filehandle)
    #     #     pickle.dump(testCase.E, filehandle)
    #     #     pickle.dump(testCase.noise_bound, filehandle)
    #     #     pickle.dump(testCase.A, filehandle)
    #     #     pickle.dump(testCase.C, filehandle)
    #     #     pickle.dump(testCase.s, filehandle)
    #     start = datetime.datetime.now()
    #     e, t = main(n, p, trial)
    #     time.append(t)
    #     error.append(e)
    #
    #     if not t:
    #         break
    #
    # print('n: {0}, error: {1}, time: {2}'.format(n, np.mean(error), np.mean(time)))
