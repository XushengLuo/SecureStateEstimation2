"""
__author__ = chrislaw
__project__ = SecureStateEstimation
__date__ = 10/5/18
"""
""" draw the figure used in paper
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14, 'legend.fontsize': 12})

# --------------------------------------- level vs iteration --------------------------------------
# with open('over2', 'rb') as filehandle:
#     level = pickle.load(filehandle)
# fig, ax = plt.subplots(2, 2)
# ax[0][0].text(27, -9.5, '(a)' )
# ax[0][0].plot(range(1, len(level) + 1), level)
# ax[0][0].set_xlabel(r'Iteration', usetex=True )
# ax[0][0].set_ylabel(r'level', usetex=True)
# ax[0][0].set_xticks([0, 20, 40, 60])
# ax[0][0].set_yticks([0, 5, 10, 15, 20])

# with open('over3', 'rb') as filehandle:
#     level = pickle.load(filehandle)
# ax[0][1].plot(range(1, len(level) + 1), level)
# ax[0][1].set_xlabel(r'Iteration', usetex=True )
# ax[0][1].set_ylabel(r'level', usetex=True)
# ax[0][1].set_yticks([0, 5, 10, 15, 20])
# ax[0][1].text(200, -9.5, '(b)' )
#
# with open('over3_r3', 'rb') as filehandle:
#     level = pickle.load(filehandle)
# ax[1][0].plot(range(1, len(level) + 1), level)
# ax[1][0].set_xlabel(r'Iteration', usetex=True )
# ax[1][0].set_ylabel(r'level', usetex=True)
# ax[1][0].set_yticks([0, 5, 10, 15, 20])
# ax[1][0].text(17, -9.5, '(c)' )
#
# with open('over3_r4', 'rb') as filehandle:
#     level = pickle.load(filehandle)
# ax[1][1].plot(range(1, len(level) + 1), level)
# ax[1][1].set_xlabel(r'Iteration', usetex=True )
# ax[1][1].set_ylabel(r'level', usetex=True)
# ax[1][1].set_yticks([0, 5, 10, 15, 20])
# ax[1][1].text(36, -9.5, '(d)' )
#
#
# fig.tight_layout()
# plt.show()

# --------------------------------------- n varies with noiseless --------------------------------------
# smt = np.array([0.154398249700000,	0.155091903300000,	0.0978938188000000,	0.190196719900000,	0.261311356100000,	0.5363312293000005])
# search = np.array([0.009015, 0.013133, 0.032277, 0.041411, 0.171899, 0.361232])
# miqp = np.array([0.05584859999999999, 0.07601880000000001, 0.0270188, 0.29438919999999996, 0.3030236, 0.2623014 ])
#
# n = np.array([10, 25, 50, 75, 100, 150])

# smt_error = np.array([3.38569233839563e-16,	1.95818672448298e-15,	3.89344813365748e-14,	4.09295088643850e-12,	3.05051986671621e-09,
#                       0.000481281063203100])
# search_error = np.array([1.2730451658146966e-15, 3.0229509803775837e-15, 7.660902400161737e-14, 3.885420093390361e-12,
#                          3.450610400660245e-09, 4.5971654934011094e-05])
#
# miqp_error = np.array([1.2730451658146966e-15,  2.3916064556910426e-15, 7.660902400161737e-14, 3.88542009339036e-12,
#                        3.19662730528689e-11, 4.597165493401109e-05])
#
#
# fig, ax = plt.subplots(2, 1)
#
# l2 = ax[0].plot(n, smt, 'r--o')
# l3 = ax[0].plot(n, miqp, 'g--s')
# l1 = ax[0].plot(n, search, 'b--d')
#
# ax[0].legend([r'IMHOTEP-SMT', r'MIQCP', r'Alg.1'])
# ax[0].set_xlabel(r'Number of states $n$', usetex=True )
# ax[0].set_ylabel(r'Execution time (sec)', usetex=True)
#
# ax[1].semilogy(n, smt_error, 'r--o')
# ax[1].semilogy(n, miqp_error, 'g--s')
# ax[1].semilogy(n, search_error, 'b--d')
#
# # plt.legend([r'Alg.1', r'IMHOTEP-SMT'])
# ax[1].set_xlabel(r'Number of states $n$', usetex=True)
# ax[1].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
#
# fig.tight_layout()
# plt.savefig('/Users/chrislaw/Box Sync/SSE/figure/vs_n.pdf', bbox_inches='tight', dpi=600)
# plt.show()

# -------------------------------- n varies with noise --------------------------------------
# smt = np.array([0.0781434985,	0.122485432 ,	0.8048681509,	2.0928267022,	2.7799694334 ])
# search = np.array([0.032514, 0.06062970000000001, 0.24615689999999998, 0.5385632, 0.9219160000000001])
# miqp = np.array([0.21515560000000003 , 0.23209439999999998, 0.1325947 , 0.1031791 , 2.3552104])
#
# n = np.array([10, 25, 50, 75, 100])
#
# smt_error = np.array([0.0651489632924267,	0.0670064026429532,	0.247245924625395,	 0.837347679516703,	11.3699118057494,
#                      ])
# search_error = np.array([0.0651489632924266, 0.06700640264295318, 0.2472459246253953, 0.837347679516696,
#                          11.369911805750544])
#
# miqp_error = np.array([0.0651489632924266,  0.06700640264295317, 0.4620594821152243, 0.8373476795166959, 11.369911805750542])
#
#
# fig, ax = plt.subplots(2, 1)
#
# l2 = ax[0].plot(n, smt, 'r--o')
# l3 = ax[0].plot(n, miqp, 'g--s')
# l1 = ax[0].plot(n, search, 'b--d')
#
# ax[0].legend([r'IMHOTEP-SMT', r'MIQCP', r'Alg.1'])
# ax[0].set_xlabel(r'Number of states $n$', usetex=True )
# ax[0].set_ylabel(r'Execution time (sec)', usetex=True)
#
# ax[1].semilogy(n, smt_error, 'r--o')
# ax[1].semilogy(n, miqp_error, 'g--s')
# ax[1].semilogy(n, search_error, 'b--d')
#
# # plt.legend([r'Alg.1', r'IMHOTEP-SMT'])
# ax[1].set_xlabel(r'Number of states $n$', usetex=True)
# ax[1].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
#
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/SSE/figure/noise_vs_n.pdf', bbox_inches='tight', dpi=600)
# plt.show()

# ----------------------------------- p varies with noiseless -------------------------
# smt = np.array([0.00517343650000000,	0.0900785035000000,	0.292310873900000,	0.587996512200000,	1.08013216610000,	2.04450678640000])
# search = np.array([0.001249, 0.017305, 0.059303, 0.09473, 0.155242, 0.271379])
# miqp = np.array([0.009306100000000001, 0.08076740000000002, 0.13222209999999998, 0.9207647999999999, 0.5926137, 3.1207153999999995 ])
#
# p = np.array([3, 30, 60, 90, 120, 150])
#
# smt_error = np.array([3.94300156107503e-13,	9.43331534203525e-16,	8.25980683692100e-16,	6.21825516131406e-16,	7.54161309646470e-16,
# 	8.14853383136167e-16])
# search_error = np.array([1.9985174439237436e-13, 3.714474161369242e-15, 1.6162963019025597e-15,
#                           3.654963358195414e-15, 2.494607697774148e-15, 3.151280538128965e-15])
#
# miqp_error = np.array([1.9985174439237436e-13, 3.714474161369242e-15, 1.6162963019025597e-15, 3.654963358195415e-15,
#                  1.3859244380905137e-15,  3.151280538128965e-15])
#
# fig, ax = plt.subplots(2, 1)
# # plt.rc('text', usetex=True)
# # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# l2 = ax[0].plot(p, smt, 'r--o')
# l3 = ax[0].plot(p, miqp, 'g--s')
# l1 = ax[0].plot(p, search, 'b--d')
# ax[0].legend([r'IMHOTEP-SMT', r'MIQCP', r'Alg.1' ])
# ax[0].set_xlabel(r'Number of sensors $p$', usetex=True)
# ax[0].set_ylabel(r'Execution time (sec)', usetex=True)
# # ax[0].set_xticks([0, 30, 60, 90, 120, 150])
#
# # ax[0].set_ylim(-0.02, 0.8)
# # ax[0].set_yticks([0,  0.2, 0.4, 0.6,  0.8])
#
# ax[1].semilogy(p, smt_error, 'r--o')
# ax[1].semilogy(p, miqp_error, 'g--s')
# ax[1].semilogy(p, search_error, 'b--d')
# ax[1].set_xlabel(r'Number of sensors $p$', usetex=True)
# ax[1].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
#
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/SSE/figure/vs_p.pdf', bbox_inches='tight', dpi=600)
# plt.show()


# ----------------------------------- p varies with noise -------------------------
# smt = np.array([0.0368, 0.1512, 0.3652, 0.7180, 1.8322])
# search = np.array([0.0246005, 0.0723677, 0.10271029999999999, 0.15601530000000002, 0.2338597])
# miqp = np.array([0.012554899999999999, 0.10243390000000001, 1.1023882999999999, 0.8653587, 0.8414181])
#
# p = np.array([30, 60, 90, 120, 150])
#
# smt_error = np.array([0.1963, 0.0964, 0.1218, 0.0450, 0.1073])
# search_error = np.array(
#     [0.19629878493085373, 0.09637933564464105, 0.12177113314118601, 0.04504335773165751, 0.10726913107924596])
#
# miqp_error = np.array(
#     [0.2096793221625214, 0.096379335644641, 0.12177113314118605, 0.04504335773165751, 0.10726913107924596])
#
# fig, ax = plt.subplots(2, 1)
# l2 = ax[0].plot(p, smt, 'r--o')
# l3 = ax[0].plot(p, miqp, 'g--s')
# l1 = ax[0].plot(p, search, 'b--d')
#
# ax[0].legend([r'IMHOTEP-SMT', r'MIQCP', r'Alg.1'])
# ax[0].set_xlabel(r'Number of sensors $p$', usetex=True)
# ax[0].set_ylabel(r'Execution time (sec)', usetex=True)
#
# ax[0].set_yticks([0, 0.5, 1.0, 1.5, 2.0])
#
# ax[1].semilogy(p, smt_error, 'r--o')
# ax[1].semilogy(p, miqp_error, 'g--s')
# ax[1].semilogy(p, search_error, 'b--d')
#
# ax[1].set_xlabel(r'Number of sensors $p$', usetex=True)
# ax[1].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
#
# # fig.legend((l1, l2), ('Alg.1', 'IMHOTEP-SMT'), 'lower center')
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/SSE/figure/noise_vs_p.pdf', bbox_inches='tight', dpi=600)
# plt.show()

# ---------------------------------- large scale test without noise ----------------------------------------------
# smt = np.array([0.046778, 0.18325, 1.8602, 7.7138, 25.6728, 70.5948, 167.6326, 352.7831, 679.7614, 1218.9168])
# search = np.array(
#     [0.0143543, 0.0906179, 0.45480280000000006, 1.3192218, 2.9070392, 6.0422193, 10.56369, 19.752612500000005,
#      28.021817799999997, 41.956478])
# miqp = np.array([0.0865253, 0.4342729, 1.8446208, 3.6280118000000003, 9.2651588, 24.7099086, 25.8255936, 47.0629264,
#                  57.547265700000004, 100.4885405])
#
# p = 20 * np.array(range(1, 11))
#
# smt_error = np.array(
#     [7.8741e-16, 2.0042e-15, 3.2185e-15, 5.1406e-15, 4.709e-15, 7.9983e-15, 8.808e-15, 9.3913e-15, 1.5132e-14,
#      1.7002e-14])
# search_error = np.array([2.3780897564033275e-15, 3.1923779247094748e-15, 3.998533829561969e-15, 5.6266140879614755e-15,
#                          5.150053079594687e-15, 6.678808820142438e-15,
#                          6.928238204182432e-15, 7.820116816245624e-15, 7.95058734750479e-15, 9.632861521347676e-15])
#
# miqp_error = np.array(
#     [2.513784316402068e-15, 1.516915605408496e-13, 5.766079505563597e-15, 0.3045216031606597, 0.10000000000654397,
#      0.10000000000000567, 0.2843618848828182, 0.17351691628932023, 0.40000000000001973, 0.4959473203884591])
#
# fig, ax = plt.subplots(2, 1)
# l2 = ax[0].plot(p, smt, 'r--o')
# l3 = ax[0].plot(p, miqp, 'g--s')
# l1 = ax[0].plot(p, search, 'b--d')
#
# ax[0].legend([r'IMHOTEP-SMT', r'MIQCP', r'Alg.1'])
# ax[0].set_xlabel(r'Number of states and sensors', usetex=True)
# ax[0].set_ylabel(r'Execution time (sec)', usetex=True)
#
# ax[0].set_xticks(p)
# ax[1].set_xticks(p)
#
# ax[1].semilogy(p, smt_error, 'r--o')
# ax[1].semilogy(p, miqp_error, 'g--s')
# ax[1].semilogy(p, search_error, 'b--d')
#
# ax[1].set_xlabel(r'Number of states and sensors', usetex=True)
# ax[1].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
#
# # fig.legend((l1, l2), ('Alg.1', 'IMHOTEP-SMT'), 'lower center')
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/Research/SSE/figure/large_scale_noiseless.pdf', bbox_inches='tight', dpi=600)
# plt.show()

# -------------------------------------- large scale test with noise --------------------------------

# smt = np.array([0.0336, 0.1180, 1.2643, 7.4214, 26.1639, 82.4168, 225.0039, 392.6660, 823.0514, 1657.7933])
# search = np.array(
#     [0.0171, 0.1080, 0.4340, 1.4479, 3.0497, 7.4229, 12.4825, 21.8938, 30.6599, 46.6422])
# miqp = np.array(
#     [0.0928741000000000, 0.758067000000000, 2.50337980000000, 4.50206960000000, 10.4537685000000, 27.6672864000000,
#      38.7556208000000, 62.2377413000000, 72.3947862, 100.5562294])
#
# p = 20 * np.array(range(1, 11))
#
# smt_error = np.array(
#     [0.0433, 0.0797, 0.1256, 0.1590, 0.1656, 0.2986, 0.2041, 0.3702, 0.2570, 0.3228])
# search_error = np.array([0.0433, 0.0797, 0.1256, 0.1590, 0.1657, 0.2986, 0.2041, 0.3702, 0.2570, 0.3228])
#
# miqp_error = np.array(
#     [0.0433464897719787, 0.119925152389902, 0.214262485811072, 1, 1, 0.446953970444272,
#      1, 0.478595482310494, 0.5727111165335478, 1])
#
# fig, ax = plt.subplots(2, 1)
# l2 = ax[0].plot(p, smt, 'r--o')
# l3 = ax[0].plot(p, miqp, 'g--s')
# l1 = ax[0].plot(p, search, 'b--d')
#
# ax[0].legend([r'IMHOTEP-SMT', r'MIQCP', r'Alg.1'])
# ax[0].set_xlabel(r'Number of states and sensors', usetex=True)
# ax[0].set_ylabel(r'Execution time (sec)', usetex=True)
#
# ax[0].set_xticks(p)
# ax[1].set_xticks(p)
#
# ax[1].plot(p, smt_error, 'r--o')
# ax[1].plot(p, miqp_error, 'g--s')
# ax[1].plot(p, search_error, 'b--d')
#
# ax[1].set_xlabel(r'Number of states and sensors', usetex=True)
# ax[1].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
#
# # fig.legend((l1, l2), ('Alg.1', 'IMHOTEP-SMT'), 'lower center')
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/Research/SSE/figure/large_scale_noise.pdf', bbox_inches='tight', dpi=600)
# plt.show()


# ------------------------------------supplemental complexity test vs # sensors --------------------------------------
# search_50_p = np.array([32.4000,64.4000,96.7000,131.9000,164.9000,199.6000,231.2000,263.0000,297.4000,333.5000])*2
# search_100_p = np.array(
#     [32.7000,64.3000,97.7000,132.0000,165.7000,197.7000,234.0000,265.5000,298.1000,330.4000])*2
# search_200_p = np.array(
#     [31.3000,62.4000,99.9000,130.9000,164.6000,196.8000,230.2000,264.8000,298.8000,332.5000])*2
#
# p = 20 * np.array(range(1, 11))
#
# fig, ax = plt.subplots(2, 1)
# l2 = ax[0].plot(p, search_50_p, 'r--o')
# l3 = ax[0].plot(p, search_100_p, 'g--s')
# l1 = ax[0].plot(p, search_200_p, 'b--d')
#
# ax[0].legend([r'n=50', r'n=100', r'n=200'], ncol=3, loc='upper center', bbox_to_anchor=(0.35, 1))
# ax[0].set_xlabel(r'Number of sensors', usetex=True)
# ax[0].set_ylabel(r'Number of nodes', usetex=True)
#
# ax[0].set_xticks(p)
# ax[0].set_yticks(200 * np.array(range(1, 4)))
# # fig.tight_layout()
#
# # plt.savefig('/Users/chrislaw/Box Sync/Research/SSE/figure/complexity_test_p.pdf', bbox_inches='tight', dpi=600)
# # plt.show()
# # # ------------------------------------supplemental complexity test vs # sensors --------------------------------------
# search_50_n = np.array([82.9000,81.3000,81.7000,82.8000,81.7000,80.7000,83.2000,82.4000,80.7000,80.4000])*2
# search_100_n = np.array(
#     [166.3000,165.2000,165.3000,165.3000,164.6000,163.4000,164.9000,165.8000,165.0000,165.2000])*2
# search_200_n = np.array(
#     [332.1000,332.6000,334.2000,330.7000,332.5000,332.2000,333.4000,332.1000,330.8000,331.1000])*2
#
# n = 20 * np.array(range(1, 11))
#
# l2 = ax[1].plot(n, search_50_n, 'r--o')
# l3 = ax[1].plot(n, search_100_n, 'g--s')
# l1 = ax[1].plot(n, search_200_n, 'b--d')
#
# ax[1].legend([r'p=50', r'p=100', r'p=200'], ncol=3, loc='upper center', bbox_to_anchor=(0.35, 0.85))
# ax[1].set_xlabel(r'Number of states', usetex=True)
# ax[1].set_ylabel(r'Number of nodes', usetex=True)
#
# ax[1].set_xticks(n)
# # ax[1].set_yticks(80 * np.array(range(1, 6)))
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/complexity_test.pdf', bbox_inches='tight', dpi=600)
# plt.show()
#
# import matplotlib.pyplot as plt
# import numpy as np

# construct some data like what you have:
# fig, ax = plt.subplots(2, 1)
# means = np.array([14.25, 15.25, 17.0, 14.25, 15.25, 17.0])
# std = np.array([1.51, 1.70, 3.10, 1.51, 1.70, 3.10])
# mins = np.array([13, 12, 13, 13, 12, 13])
# maxes = np.array([17, 18, 27, 17, 18, 27])
# worst = np.array([26, 19, 29, 26, 19, 29])
#
# # create stacked errorbars:
# ax[0].errorbar(np.array([2,3,4]), means[range(3)], std[range(3)], fmt='.c', ecolor='c', lw=15, label='constant attack')
# ax[0].errorbar(np.array([5,6,7]), means[range(3, 6)], std[range(3, 6)], fmt='.g', ecolor='g', lw=15, label='random attack')
#
# ax[0].errorbar(np.array([2,3,4]), means[range(3)], [means[range(3)] - mins[range(3)],
#                                                      maxes[range(3)] - means[range(3)]],
#              fmt='.k', markersize=8, ecolor='k', lw=1, capsize=5, capthick=1)
# ax[0].errorbar(np.array([5,6,7]), means[range(3, 6)], [means[range(3, 6)] - mins[range(3, 6)],
#                                                      maxes[range(3, 6)] - means[range(3, 6)]],
#              fmt='.k', markersize=8, ecolor='k', lw=1, capsize=5, capthick=1)
# # plt.plot(np.arange(8), means+3, '*k')
# # plt.xlim(-1, 8)
# ax[0].plot([2,3,4,5,6,7], worst, '*r', markersize=10, label='appear first')
# ax[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.3))
#
# ax[0].set_yticks([15, 20, 25, 30])
# ax[0].set_xticks(np.array([2, 3, 4, 5, 6, 7]))
# ax[0].set_xticklabels([2, 3, 4, 2, 3, 4])
# ax[0].set_ylabel('iterations')
# ax[0].set_title('(a) Noiseless secure state estimation', y=-0.35)
# # fig.tight_layout()
#
# means = np.array([14.85, 17.15, 18.25, 14.85, 17.25, 18.25])
# std = np.array([2.03, 3.62, 4.84, 2.03, 3.81, 4.84])
# mins = np.array([13, 13, 13, 13, 13, 13])
# maxes = np.array([19, 29, 29, 19, 29, 29])
# worst = [22, 22, 29, 22, 22, 29]
#
# # create stacked errorbars:
# ax[1].errorbar(np.array([2,3,4]), means[range(3)], std[range(3)], fmt='.c', ecolor='c', lw=15, label='constant attack')
# ax[1].errorbar(np.array([5,6,7]), means[range(3, 6)], std[range(3, 6)], fmt='.g', ecolor='g', lw=15, label='random attack')
#
# ax[1].errorbar(np.array([2,3,4]), means[range(3)], [means[range(3)] - mins[range(3)],
#                                                      maxes[range(3)] - means[range(3)]],
#              fmt='.k', markersize=8, ecolor='k', lw=1, capsize=5, capthick=1)
# ax[1].errorbar(np.array([5,6,7]), means[range(3, 6)], [means[range(3, 6)] - mins[range(3, 6)],
#                                                      maxes[range(3, 6)] - means[range(3, 6)]],
#              fmt='.k', markersize=8, ecolor='k', lw=1, capsize=5, capthick=1)
# # plt.plot(np.arange(8), means+3, '*k')
# # plt.xlim(-1, 8)
# ax[1].plot([2,3,4,5,6,7], worst, '*r', markersize=10, label='appear first')
# # ax[1].axvline(4.5, which='major', linestyle='-.', color='gray')  # vertical lines
#
# ax[1].set_xlabel(r'$\bar{s}$')
#
# ax[1].set_yticks([15, 20, 25, 30])
# ax[1].set_xticks(np.array([2,3,4,5,6,7]))
# ax[1].set_xticklabels([2,3,4,2,3,4])
# # ax[1].set_xlabel(r'$\bar{s}$')
# ax[1].set_ylabel('iterations')
# ax[1].set_title('(b) Noisy secure state estimation', y=-0.5)
# plt.show()


# 2, noiseless, worst, 14.22, 5.039007838850819, 11, 26, 1.9728977446515544e-15
# 3, noiseless, worst, 21.8, 5.979966555090421, 11, 32, 3.2422226449230927e-15
# 4, noiseless, worst, 28.25, 3.766629793329841, 22, 38, 2.0446764135867933e-15
# 2, noiseless, random, 13.9075, 2.256976683530426, 11, 26, 2.025872546801118e-15
# 3, noiseless, random, 15.79, 2.4384216206390557, 11, 32, 2.761354484923665e-15
# 4, noiseless, random, 17.33, 4.1335335972990475, 13, 38, 2.265020539311472e-15
# 2, noisy, worst, 13.9175, 4.884740909198768, 11, 26, 0.09276686005919153
# 3, noisy, worst, 19.7525, 5.128473822688383, 11, 30, 0.09158693799925921
# 4, noisy, worst, 26.8075, 2.910402678324771, 22, 33, 0.19713239617726985
# 2, noisy, random, 14.0, 2.303258561256204, 11, 22, 0.09688393937079379
# 3, noisy, random, 15.7475, 3.3269721594867607, 11, 47, 0.11444131522683518
# 4, noisy, random, 17.2775, 4.179771973445441, 12, 42, 0.18553701119517657

# ----------------- small system -------------------
# import pickle
#
# with open('result_small_iter', 'rb') as filehandle:
#     result = pickle.load(filehandle)
#
# fig, ax = plt.subplots(1, 2)
# means = result[range(6), 0]
# std = result[range(6), 1]
# mins = result[range(6), 2]
# maxes = result[range(6), 3]
#
# # create stacked errorbars:
# ax[0].errorbar(np.array([2, 3, 4]), means[range(3, 6)], std[range(3, 6)], fmt='.c', ecolor='c', lw=17,
#                label='random attack', alpha=0.5)
# ax[0].errorbar(np.array([2.3, 3.3, 4.3]), means[range(3)], std[range(3)], fmt='.m', ecolor='m', lw=17,
#                label='greedy attack', alpha=0.5)
#
# ax[0].errorbar(np.array([2.3, 3.3, 4.3]), means[range(3)], [means[range(3)] - mins[range(3)],
#                                                      maxes[range(3)] - means[range(3)]],
#              fmt='.k', markersize=8, ecolor='m', lw=1, capsize=5, capthick=1)
# ax[0].errorbar(np.array([2, 3, 4]), means[range(3, 6)], [means[range(3, 6)] - mins[range(3, 6)],
#                                                      maxes[range(3, 6)] - means[range(3, 6)]],
#              fmt='.k', markersize=8, ecolor='c', lw=1, capsize=5, capthick=1)
# # plt.plot(np.arange(8), means+3, '*k')
# # ax[0].plot([2,3,4,5,6,7], worst, '*r', markersize=10, label='appear first')
# ax[0].legend(ncol=1, bbox_to_anchor=(0.8, 1.0))
# ax[0].set_xlabel(r'$\bar{s}$')
#
# ax[0].set_yticks([10, 20, 30, 40, 50, 60])
#
# ax[0].set_xticks(np.array([1.8, 2.15, 3.15, 4.15, 4.5]))
# ax[0].set_xticklabels([None, 2, 3,  4])
# ax[0].set_ylabel('Number of iterations')
# ax[0].set_title('(a) Noiseless case', y=-0.25)
# # fig.tight_layout()
#
# means = result[range(6, 12), 0]
# std = result[range(6, 12), 1]
# mins = result[range(6, 12), 2]
# maxes = result[range(6, 12), 3]
#
# # create stacked errorbars:
# ax[1].errorbar(np.array([2, 3, 4]), means[range(3, 6)], std[range(3, 6)], fmt='.c', ecolor='c', lw=17,
#                label='random attack', alpha=0.5)
#
# ax[1].errorbar(np.array([2.3, 3.3, 4.3]), means[range(3)], std[range(3)], fmt='.m', ecolor='m', lw=17,
#                label='greedy attack', alpha=0.5)
#
# ax[1].errorbar(np.array([2, 3, 4]), means[range(3, 6)], [means[range(3, 6)] - mins[range(3, 6)],
#                                                      maxes[range(3, 6)] - means[range(3, 6)]],
#              fmt='.k', markersize=8, ecolor='c', lw=1, capsize=5, capthick=1)
#
# ax[1].errorbar(np.array([2.3, 3.3, 4.3]), means[range(3)], [means[range(3)] - mins[range(3)],
#                                                      maxes[range(3)] - means[range(3)]],
#              fmt='.k', markersize=8, ecolor='m', lw=1, capsize=5, capthick=1)
#
# # plt.plot(np.arange(8), means+3, '*k')
# # plt.xlim(-1, 8)
# # ax[1].plot([2,3,4,5,6,7], worst, '*r', markersize=10, label='appear first')
# # ax[1].axvline(4.5, which='major', linestyle='-.', color='gray')  # vertical lines
#
# ax[1].set_xlabel(r'$\bar{s}$')
# ax[1].set_yticks([10, 20, 30, 40, 50, 60])
#
# # ax[1].set_xticks(np.array([2,3,4,5,6,7]))
# # ax[1].set_xticklabels([2,3,4,2,3,4])
# # ax[1].set_xlabel(r'$\bar{s}$')
# ax[1].set_xticks(np.array([1.8, 2.15, 3.15, 4.15, 4.5]))
# ax[1].set_xticklabels([None, 2, 3,  4])
# # ax[1].set_ylabel('iterations')
# ax[1].set_title('(b) Noisy case', y=-0.25)
# plt.subplots_adjust(bottom=0.18, top=0.91, wspace=0.25)
# plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/small_noisy.pdf', bbox_inches='tight', dpi=600)
# plt.show()


# --------------------- scalability ---------------
import pickle
import scipy.io as sio

a = ['a', 'b', 'c']
n = [10, 20, 30]

for noise in ['noiseless', 'noisy']:
    fig, ax_itera = plt.subplots(1, 3)
    for percent in [1, 2, 3]:
        sensor = [[], []]
        means = [[], []]
        std = [[], []]
        mins = [[], []]
        maxes = [[], []]

        means_itera = [[], []]
        std_itera = [[], []]
        mins_itera = [[], []]
        maxes_itera = [[], []]

        sensor_ml = [[], []]
        means_ml = [[], []]
        std_ml = [[], []]
        mins_ml = [[], []]
        maxes_ml = [[], []]

        means_error = [[], []]
        std_error = [[], []]
        mins_error = [[], []]
        maxes_error = [[], []]

        mistake_ratio = [[], []]

        for p in range(1, 11):
            with open('result/{0}_{1}_{2}_{3}.mat'.format(noise, 'random', p*20, percent), 'rb') as filehandle:
                time = pickle.load(filehandle)
                error = pickle.load(filehandle)
                itera = (np.array(pickle.load(filehandle)))
                mistake = pickle.load(filehandle)
            sensor[0].append(p*20)
            means[0].append(np.mean(time))
            std[0].append(np.std(time))
            mins[0].append(np.min(time))
            maxes[0].append(np.max(time))

            means_itera[0].append(np.mean(itera))
            std_itera[0].append(np.std(itera))
            mins_itera[0].append(np.min(itera))
            maxes_itera[0].append(np.max(itera))

            means_error[0].append(np.mean(error))
            std_error[0].append(np.std(error))
            mins_error[0].append(np.min(error))
            maxes_error[0].append(np.max(error))

            mistake_ratio[0].append(len(mistake)/25)

            data = sio.loadmat('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/'
                            'Random examples/{0}_{1}_{2}_{3}.mat'.format(noise, 'random', p*20, percent))
            time_ml = data['TimeSpent']
            error_ml = data['error']
            mistake_ml = data['mistake']

            sensor_ml[0].append(p*20)
            means_ml[0].append(np.mean(time_ml))
            std_ml[0].append(np.std(time_ml))
            mins_ml[0].append(np.min(time_ml))
            maxes_ml[0].append(np.max(time_ml))

            if p == 10:
                print(noise, percent, np.mean(time), np.mean(time_ml))

            with open('result/{0}_{1}_{2}_{3}.mat'.format(noise, 'worst', p*20, percent), 'rb') as filehandle:
                time = pickle.load(filehandle)
                error = pickle.load(filehandle)
                itera = (np.array(pickle.load(filehandle)))
                mistake = pickle.load(filehandle)

            sensor[1].append(p*20)
            means[1].append(np.mean(time))
            std[1].append(np.std(time))
            mins[1].append(np.min(time))
            maxes[1].append(np.max(time))

            means_itera[1].append(np.mean(itera))
            std_itera[1].append(np.std(itera))
            mins_itera[1].append(np.min(itera))
            maxes_itera[1].append(np.max(itera))

            means_error[1].append(np.mean(error))
            std_error[1].append(np.std(error))
            mins_error[1].append(np.min(error))
            maxes_error[1].append(np.max(error))

            mistake_ratio[1].append(len(mistake)/25)


            data = sio.loadmat(
                '/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/'
                'Random examples/{0}_{1}_{2}_{3}.mat'.format(noise, 'worst', p * 20, percent))
            time_ml = data['TimeSpent']
            error_ml = data['error']
            mistake_ml = data['mistake']

            sensor_ml[1].append(p*20)
            means_ml[1].append(np.mean(time_ml))
            std_ml[1].append(np.std(time_ml))
            mins_ml[1].append(np.min(time_ml))
            maxes_ml[1].append(np.max(time_ml))

        # --------------- runtime  -----------------
        # fig, ax = plt.subplots(1, 2)
        #
        # # create stacked errorbars:
        # ax[0].errorbar(sensor[0], means[0], std[0], fmt='.c', ecolor='c', lw=15, label='Alg.1', alpha=0.3)
        #
        # ax[0].errorbar(sensor[0], means[0], [np.array(means[0]) - np.array(mins[0]),
        #                                      np.array(maxes[0]) - np.array(means[0])],
        #              fmt='.k', markersize=5, ecolor='c', lw=1, capsize=5, capthick=1)
        # ax[0].plot(sensor[0], means[0], '-c', markersize=1)
        #
        # ax[0].errorbar(sensor_ml[0], means_ml[0], std_ml[0], fmt='.m', ecolor='m', lw=15, label='IMHOTEP-SMT', alpha=0.3)
        #
        # ax[0].errorbar(sensor_ml[0], means_ml[0], [np.array(means_ml[0]) - np.array(mins_ml[0]),
        #                                      np.array(maxes_ml[0]) - np.array(means_ml[0])],
        #              fmt='.k', markersize=5, ecolor='m', lw=1, capsize=5, capthick=1)
        # ax[0].plot(sensor_ml[0], means_ml[0], '-m', markersize=1)
        #
        # ax[0].legend(ncol=1, bbox_to_anchor=(0.87, 1.0))
        # ax[0].set_xlabel(r'${p=n}$')
        # ax[0].set_xticks(np.array([0, 40, 80, 120, 160, 200]))
        #
        # ax[0].set_yticks(np.array(range(0, int(np.max(maxes_ml[0])//200+2)))*200)
        #
        # # # ax[0].set_xticklabels([None, 2, 3,  4])
        # ax[0].set_ylabel('Time(sec)')
        # ax[0].set_title('Random attack', y=-0.23)
        #
        # ax[1].errorbar(sensor[1], means[1], std[1], fmt='.c', ecolor='c', lw=15, alpha=0.3)
        # ax[1].errorbar(sensor[1], means[1], [np.array(means[1]) - np.array(mins[1]),
        #                                                np.array(maxes[1]) - np.array(means[1])],
        #              fmt='.k', markersize=5, ecolor='c', lw=1, capsize=5, capthick=1)
        # ax[1].plot(sensor[1], means[1], '-c', markersize=1)
        #
        # ax[1].errorbar(sensor_ml[1], means_ml[1], std_ml[1], fmt='.m', ecolor='m', lw=15, alpha=0.3)
        #
        # ax[1].errorbar(sensor_ml[1], means_ml[1], [np.array(means_ml[1]) - np.array(mins_ml[1]),
        #                                      np.array(maxes_ml[1]) - np.array(means_ml[1])],
        #              fmt='.k', markersize=5, ecolor='m', lw=1, capsize=5, capthick=1)
        # ax[1].plot(sensor_ml[1], means_ml[1], '-m', markersize=1)
        #
        # ax[1].set_xlabel(r'${p=n}$')
        # ax[1].set_xticks(np.array([0, 40, 80, 120, 160, 200]))
        # ax[1].set_yticks(np.array(range(0, int(np.max(maxes_ml[0])//200+2)))*200)
        # ax[1].set_title('Greedy attack', y=-0.23)
        #
        # plt.subplots_adjust(bottom=0.17, top=0.9, wspace=0.25)
        #
        # plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/{0}_{1}.pdf'.format(noise, percent), bbox_inches='tight', dpi=600)

    #     ------------- number of iterations ------------------
        ax_itera[percent-1].errorbar(sensor[0], means_itera[0], std_itera[0], fmt='.c', ecolor='c', lw=10,
                                     label='random attack', alpha=0.3)

        ax_itera[percent-1].errorbar(sensor[0], means_itera[0], [np.array(means_itera[0]) - np.array(mins_itera[0]),
                                             np.array(maxes_itera[0]) - np.array(means_itera[0])],
                     fmt='.k', markersize=5, ecolor='c', lw=1, capsize=5, capthick=1)
        ax_itera[percent-1].plot(sensor[0], means_itera[0], '-c', markersize=1)

        ax_itera[percent-1].set_xticks(np.array([0, 40, 120, 200]))
        # ax_itera[percent-1].set_xlabel(r'${p=n}$')
        ax_itera[percent-1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        ax_itera[percent-1].errorbar(sensor[1], means_itera[1], std_itera[1], fmt='.m', ecolor='m', lw=10,
                                     label='greedy attack', alpha=0.3)
        ax_itera[percent-1].errorbar(sensor[1], means_itera[1], [np.array(means_itera[1]) - np.array(mins_itera[1]),
                                                       np.array(maxes_itera[1]) - np.array(means_itera[1])],
                     fmt='.k', markersize=5, ecolor='m', lw=1, capsize=5, capthick=1)
        ax_itera[percent-1].plot(sensor[1], means_itera[1], '-m', markersize=1)
        ax_itera[percent-1].set_title(r'({0}) $s/p={1}\%$'.format(a[percent - 1], n[percent - 1]), y=-0.18)
        ax_itera[percent-1].set_yticks(np.array(range(0, 5))*100)

    ax_itera[0].set_ylabel(r'Number of iterations')
    ax_itera[0].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.6, 1.2))

    plt.subplots_adjust(bottom=0.15, top=0.85)
    plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/iter_{0}.pdf'.format(noise), bbox_inches='tight', dpi=600)
    #
    # ------------- accuracy ------------------
    #     ax_itera[percent - 1].errorbar(sensor[0], means_error[0], std_error[0], fmt='.c', ecolor='c', lw=5,
    #                                    label='random attack', alpha=0.3)
    #
    #     ax_itera[percent - 1].errorbar(sensor[0], means_error[0], [np.array(means_error[0]) - np.array(mins_error[0]),
    #                                                                np.array(maxes_error[0]) - np.array(means_error[0])],
    #                                    fmt='.k', markersize=5, ecolor='c', lw=1, capsize=5, capthick=1)
    #     ax_itera[percent - 1].plot(sensor[0], means_error[0], '-c', markersize=1)
    #
    #     ax_itera[percent - 1].set_xticks(np.array([0, 40, 120, 200]))
    #     # ax_itera[percent-1].set_xlabel(r'${p=n}$')
    #     # ax_itera[percent - 1].ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
    #
    #     ax_itera[percent - 1].errorbar(sensor[1], means_error[1], std_error[1], fmt='.m', ecolor='m', lw=5,
    #                                    label='greedy attack', alpha=0.3)
    #     ax_itera[percent - 1].errorbar(sensor[1], means_error[1], [np.array(means_error[1]) - np.array(mins_error[1]),
    #                                                                np.array(maxes_error[1]) - np.array(means_error[1])],
    #                                    fmt='.k', markersize=5, ecolor='m', lw=1, capsize=5, capthick=1)
    #     ax_itera[percent - 1].plot(sensor[1], means_error[1], '-m', markersize=1)
    #     ax_itera[percent - 1].set_title(r'({0}) $s/p={1}\%$'.format(a[percent - 1], n[percent - 1]), y=-0.18)
    #     # ax_itera[percent - 1].set_yticks(np.array(range(0, 9)) * 100)
    #
    #
    #
    # # ax_itera[1].set_xticks(np.array([0, 40, 120, 200]))
    #     # ax_itera[1].set_xlabel(r'${p=n}$')
    #     # ax_itera[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax_itera[0].set_ylabel(r'$\|x^* - x_0\|_2 / \|x_0\|_2$', usetex=True)
    # ax_itera[0].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.6, 1.2))
    #
    # plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.27)
    # plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/error_{0}.pdf'.format(noise), bbox_inches='tight', dpi=600)
plt.show()
    # for percent in [1]:
    #     for p in range(1, 11):






# search_50_p = np.array([32.4000,64.4000,96.7000,131.9000,164.9000,199.6000,231.2000,263.0000,297.4000,333.5000])*2
# search_100_p = np.array(
#     [32.7000,64.3000,97.7000,132.0000,165.7000,197.7000,234.0000,265.5000,298.1000,330.4000])*2
# search_200_p = np.array(
#     [31.3000,62.4000,99.9000,130.9000,164.6000,196.8000,230.2000,264.8000,298.8000,332.5000])*2
#
# p = 20 * np.array(range(1, 11))
#
# fig, ax = plt.subplots(2, 1)
# l2 = ax[0].plot(p, search_50_p, 'r--o')
# l3 = ax[0].plot(p, search_100_p, 'g--s')
# l1 = ax[0].plot(p, search_200_p, 'b--d')
#
# ax[0].legend([r'n=50', r'n=100', r'n=200'], ncol=3, loc='upper center', bbox_to_anchor=(0.35, 1))
# ax[0].set_xlabel(r'Number of sensors', usetex=True)
# ax[0].set_ylabel(r'Number of nodes', usetex=True)
#
# ax[0].set_xticks(p)
# ax[0].set_yticks(200 * np.array(range(1, 4)))
# # fig.tight_layout()
#
# # plt.savefig('/Users/chrislaw/Box Sync/Research/SSE/figure/complexity_test_p.pdf', bbox_inches='tight', dpi=600)
# # plt.show()
# # # ------------------------------------supplemental complexity test vs # sensors --------------------------------------
# search_50_n = np.array([82.9000,81.3000,81.7000,82.8000,81.7000,80.7000,83.2000,82.4000,80.7000,80.4000])*2
# search_100_n = np.array(
#     [166.3000,165.2000,165.3000,165.3000,164.6000,163.4000,164.9000,165.8000,165.0000,165.2000])*2
# search_200_n = np.array(
#     [332.1000,332.6000,334.2000,330.7000,332.5000,332.2000,333.4000,332.1000,330.8000,331.1000])*2
#
# n = 20 * np.array(range(1, 11))
#
# l2 = ax[1].plot(n, search_50_n, 'r--o')
# l3 = ax[1].plot(n, search_100_n, 'g--s')
# l1 = ax[1].plot(n, search_200_n, 'b--d')
#
# ax[1].legend([r'p=50', r'p=100', r'p=200'], ncol=3, loc='upper center', bbox_to_anchor=(0.35, 0.85))
# ax[1].set_xlabel(r'Number of states', usetex=True)
# ax[1].set_ylabel(r'Number of nodes', usetex=True)
#
# ax[1].set_xticks(n)
# # ax[1].set_yticks(80 * np.array(range(1, 6)))
# fig.tight_layout()
#
# plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/complexity_test.pdf', bbox_inches='tight', dpi=600)
# plt.show()

# -------------- supplementary complexity  -----------------
# import pickle
# import scipy.io as sio
#
# a = ['g', 'c', 'm']
# n = [10, 20, 30]
#
# fig, ax_itera = plt.subplots(2, 1)
# for noise in ['noiseless']:
#     for percent in [3]:
#         for index, n in enumerate([50, 100, 200]):
#             sensor = [[], []]
#             means_itera = [[], []]
#             std_itera = [[], []]
#             mins_itera = [[], []]
#             maxes_itera = [[], []]
#             for p in range(1, 11):
#                 with open('result/complexity_{0}_{1}_{2}_{3}_{4}.mat'.format('noiseless', 'random', p*20, n, 3), 'rb') as filehandle:
#                     time = pickle.load(filehandle)
#                     error = pickle.load(filehandle)
#                     itera = (np.array(pickle.load(filehandle)))
#                     mistake = pickle.load(filehandle)
#
#                 sensor[0].append(p*20)
#                 means_itera[0].append(np.mean(itera))
#                 std_itera[0].append(np.std(itera))
#                 mins_itera[0].append(np.min(itera))
#                 maxes_itera[0].append(np.max(itera))
#
#             ax_itera[0].errorbar(sensor[0], means_itera[0], std_itera[0], fmt='.{0}'.format(a[index]),
#                                  ecolor='{0}'.format(a[index]), lw=10,
#                                  label=r'$n={0}$'.format(n), alpha=0.3)
#
#             ax_itera[0].errorbar(sensor[0], means_itera[0], [np.array(means_itera[0]) - np.array(mins_itera[0]),
#                                                              np.array(maxes_itera[0]) - np.array(means_itera[0])],
#                                  fmt='.k', markersize=5, ecolor='{0}'.format(a[index]), lw=1, capsize=5, capthick=1)
#             ax_itera[0].plot(sensor[0], means_itera[0], '-{0}'.format(a[index]), markersize=1)
#
#             ax_itera[0].set_xticks(np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]))
#             ax_itera[0].set_xlabel(r'${p}$')
#             # ax_itera[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#
#             ax_itera[0].set_ylabel(r'Number of iterations')
#
#             # plt.subplots_adjust(bottom=0.15, top=0.85)
#
#         ax_itera[0].legend(ncol=1, loc='upper center', bbox_to_anchor=(0.15, 1))
#
#         for index, p in enumerate([50, 100, 200]):
#             sensor = [[], []]
#             means_itera = [[], []]
#             std_itera = [[], []]
#             mins_itera = [[], []]
#             maxes_itera = [[], []]
#             for n in range(1, 11):
#                 with open('result/pcomplexity_{0}_{1}_{2}_{3}_{4}.mat'.format('noiseless', 'random', p, n*20, 3), 'rb') as filehandle:
#                     time = pickle.load(filehandle)
#                     error = pickle.load(filehandle)
#                     itera = (np.array(pickle.load(filehandle)))
#                     mistake = pickle.load(filehandle)
#
#                 sensor[0].append(n*20)
#                 means_itera[0].append(np.mean(itera))
#                 std_itera[0].append(np.std(itera))
#                 mins_itera[0].append(np.min(itera))
#                 maxes_itera[0].append(np.max(itera))
#
#             ax_itera[1].errorbar(sensor[0], means_itera[0], std_itera[0], fmt='.{0}'.format(a[index]),
#                                  ecolor='{0}'.format(a[index]), lw=10,
#                                  label=r'$p={0}$'.format(p), alpha=0.3)
#
#             ax_itera[1].errorbar(sensor[0], means_itera[0], [np.array(means_itera[0]) - np.array(mins_itera[0]),
#                                                              np.array(maxes_itera[0]) - np.array(means_itera[0])],
#                                  fmt='.k', markersize=5, ecolor='{0}'.format(a[index]), lw=1, capsize=5, capthick=1)
#             ax_itera[1].plot(sensor[0], means_itera[0], '-{0}'.format(a[index]), markersize=1)
#
#             ax_itera[1].set_xticks(np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]))
#             ax_itera[1].set_xlabel(r'${n}$')
#             # ax_itera[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#
#             ax_itera[1].set_ylabel(r'Number of iterations')
#         ax_itera[1].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.4, 0.85))
#         plt.subplots_adjust(top=0.92, hspace=0.37)
#         plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/complexity_test.pdf', bbox_inches='tight', dpi=600)
#
# plt.show()

# -------------- supplementary complexity runtime -----------------
# a = ['g', 'c', 'm']
# n = [10, 20, 30]
#
# fig, ax_time = plt.subplots(2, 1)
# for noise in ['noiseless']:
#     for percent in [3]:
#         for index, n in enumerate([50, 100, 200]):
#             sensor = [[], []]
#             means_time = [[], []]
#             std_time = [[], []]
#             mins_time = [[], []]
#             maxes_time = [[], []]
#             for p in range(1, 11):
#                 with open('result/complexity_{0}_{1}_{2}_{3}_{4}.mat'.format('noiseless', 'random', p*20, n, 3), 'rb') as filehandle:
#                     time = pickle.load(filehandle)
#                     error = pickle.load(filehandle)
#                     itera = (np.array(pickle.load(filehandle)))
#                     mistake = pickle.load(filehandle)
#
#                 sensor[0].append(p*20)
#                 means_time[0].append(np.mean(time))
#                 std_time[0].append(np.std(time))
#                 mins_time[0].append(np.min(time))
#                 maxes_time[0].append(np.max(time))
#
#             ax_time[0].errorbar(sensor[0], means_time[0], std_time[0], fmt='.{0}'.format(a[index]),
#                                  ecolor='{0}'.format(a[index]), lw=10,
#                                  label=r'$n={0}$'.format(n), alpha=0.3)
#
#             ax_time[0].errorbar(sensor[0], means_time[0], [np.array(means_time[0]) - np.array(mins_time[0]),
#                                                              np.array(maxes_time[0]) - np.array(means_time[0])],
#                                  fmt='.k', markersize=5, ecolor='{0}'.format(a[index]), lw=1, capsize=5, capthick=1)
#             ax_time[0].plot(sensor[0], means_time[0], '-{0}'.format(a[index]), markersize=1)
#
#             ax_time[0].set_xticks(np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]))
#             ax_time[0].set_xlabel(r'${p}$')
#             # ax_time[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#             ax_time[0].set_yticks(np.array([0, 10, 20, 30, 40]))
#             ax_time[0].set_ylabel(r'Time(sec)')
#
#             # plt.subplots_adjust(bottom=0.15, top=0.85)
#
#         ax_time[0].legend(ncol=1, loc='upper center', bbox_to_anchor=(0.15, 1))
#
#         for index, p in enumerate([50, 100, 200]):
#             sensor = [[], []]
#             means_time = [[], []]
#             std_time = [[], []]
#             mins_time = [[], []]
#             maxes_time = [[], []]
#             for n in range(1, 11):
#                 with open('result/pcomplexity_{0}_{1}_{2}_{3}_{4}.mat'.format('noiseless', 'random', p, n*20, 3), 'rb') as filehandle:
#                     time = pickle.load(filehandle)
#                     error = pickle.load(filehandle)
#                     itera = (np.array(pickle.load(filehandle)))
#                     mistake = pickle.load(filehandle)
#
#                 sensor[0].append(n*20)
#                 means_time[0].append(np.mean(time))
#                 std_time[0].append(np.std(time))
#                 mins_time[0].append(np.min(time))
#                 maxes_time[0].append(np.max(time))
#
#             ax_time[1].errorbar(sensor[0], means_time[0], std_time[0], fmt='.{0}'.format(a[index]),
#                                  ecolor='{0}'.format(a[index]), lw=10,
#                                  label=r'$p={0}$'.format(p), alpha=0.3)
#
#             ax_time[1].errorbar(sensor[0], means_time[0], [np.array(means_time[0]) - np.array(mins_time[0]),
#                                                              np.array(maxes_time[0]) - np.array(means_time[0])],
#                                  fmt='.k', markersize=5, ecolor='{0}'.format(a[index]), lw=1, capsize=5, capthick=1)
#             ax_time[1].plot(sensor[0], means_time[0], '-{0}'.format(a[index]), markersize=1)
#
#             ax_time[1].set_xticks(np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]))
#             ax_time[1].set_xlabel(r'${n}$')
#             # ax_time[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#             ax_time[1].set_yticks(np.array([0, 10, 20, 30, 40]))
#
#             ax_time[1].set_ylabel(r'Time(sec)')
#         ax_time[1].legend(ncol=1, loc='upper center', bbox_to_anchor=(0.15, 1))
#         plt.subplots_adjust(top=0.92, hspace=0.37)
#         plt.savefig('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/figure/complexity_test_time.pdf', bbox_inches='tight', dpi=600)
#
# plt.show()