# -*- coding: utf-8 -*-  # Λ
# import IPython; IPython.get_ipython().magic('reset -sf')
import os, sys, copy, time, random, cv2, matplotlib
# matplotlib.use('QT4Agg')
import numpy as np, cvxpy as cv, matplotlib.pyplot as pl, pylab as py
from mpl_toolkits.mplot3d import Axes3D

tt = time.time()
print('\n'*100)
np.set_printoptions(precision=3)

# print('\ntime elapsed=', time.time()-tt)
# exec(open('file.py').read())
# newArray = np.array([[0,1,2], [3,4,5], [6,7,8]])
# sliceArray = PTl[np.asmatrix([1,3,5]).T, [1,3,5]]
# pl.figure(figsize=(6, 6))
# fig = pl.figure()
# ax = fig.gca(projection='3d')
# ax.plot(xs=p.value[0,:].A1,ys=p.value[1,:].A1,zs=p.value[2,:].A1)
# legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), prop={'size': 18})
# print(dir(ct))
# raise Exception('exit')
# print(cv.installed_solvers())

import quad_metric_data as ct

P = cv.Semidef(ct.n)  # cv.Variable(ct.n, ct.n)  # slower


def OBJ(X, Y, n_, N, d_, P):
    U = cv.Parameter(n_, N)  # *X.shape
    UUT = cv.Parameter(n_, n_)
    dd = cv.Parameter(N, sign="positive")
    # THIS AFFECTS CONVEXITY!!!!!!
    # DIDN'T KNOW UNTIL ACCIDENTALLY MADE dd A PARAMETER

    U.value = X-Y
    UUT.value = np.dot(X-Y, (X-Y).T)
    dd.value = d_

    term1 = sum(d_**2)
    # term2 = sum([dd[i]*cv.sqrt(cv.quad_form(U[:, i], P)) for i in range(N)])
    term2 = dd.T * cv.sqrt(cv.diag(U.T*P*U))
    term3 = cv.trace(UUT * P)
    obj = 1/N * cv.Minimize(term1 - 2*term2 + term3)
    return obj

obj = OBJ(ct.X, ct.Y, ct.n, ct.N, ct.d, P)
prb = cv.Problem(obj, [P >> 0, P == P.T])

prb.solve(solver='SCS', max_iters=10000)  # , verbose=True)
print(prb.status)
print('mean squared distance error, training', prb.value)

###########

obj = OBJ(ct.X_test, ct.Y_test, ct.n, ct.N_test, ct.d_test, P)

print('mean squared distance error, testing', obj.value)
