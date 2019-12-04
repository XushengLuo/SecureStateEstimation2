from itertools import combinations
import numpy as np


def complexity(p, max_s):
    S = p - 2 * max_s
    s = int(p * 0.1) # max_s
    it = 0
    for i in range(1, S+1):
        if s >= i:
            # print(len(list(combinations(range(s), i))), len(list(combinations(range(max_s+S-s), S-i))), (max_s + S))
            it += len(list(combinations(range(s), i))) * len(list(combinations(range(max_s+S-s), S-i))) * (max_s + S)
    it += p
    return it


print(complexity(200, 96))
#     s = int(p * 0.1)
# 200 1 39801
# 200 2 3861794
# 200 3 247213530
# 200 99 183010
# 200 98 255052730
# 200 97 108396498461



# ----------------- propertiew of LTI system --------------------------
# import pickle
# for def_s in [2, 3, 4]:
#     delta = []
#     for trial in range(1, 21):
#         with open('smalldata/{7}small_{6}_sse_test_from_mat_n{0}_p{1}_{2}_{3}_{4}_{5}.mat'.format(10, 10, trial, 1, 'random', 'random', 'noiseless', def_s), 'rb') as filehandle:
#             Y = pickle.load(filehandle)
#             obsMatrix = pickle.load(filehandle)
#             p, n, tau = pickle.load(filehandle)
#             K = pickle.load(filehandle)
#             x0 = pickle.load(filehandle)
#             E = pickle.load(filehandle)
#             noise_bound = pickle.load(filehandle)
#             A = pickle.load(filehandle)
#             C = pickle.load(filehandle)
#             tol = pickle.load(filehandle)
#             s = pickle.load(filehandle)
#             max_s = pickle.load(filehandle)
#             delta.append(pickle.load(filehandle))
#     print(def_s, np.mean(delta), np.std(delta), np.min(delta), np.max(delta),  2/np.sqrt(1-np.mean(delta)))