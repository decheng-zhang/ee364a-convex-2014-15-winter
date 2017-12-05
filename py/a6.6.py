from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import cvxpy as cv
import matplotlib as mp
# mp.use('QT4Agg')
import matplotlib.pyplot as pl
import pylab as py
import os, sys, time, copy#Λ
# haha=np.array([[0,1,2],
#                [3,4,5],
#                [6,7,8]])
print('\n'*100)
tt = time.time()
# print('\ntime elapsed=', time.time()-tt)
# exec(open('ls_perm_meas_data.py').read())
import ml_estim_incr_signal_data as ct
np.set_printoptions(precision=3)
# PTl[np.asmatrix([1,3,5]).T, [1,3,5]]
# pl.figure(figsize=(6, 6))
# legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), prop={'size': 18})
# print(dir(ct))

H = np.zeros((ct.N, ct.N))
for idx in np.arange(ct.h.size):
    Htemp = np.diag(np.ones(ct.N) * ct.h[idx])
    Htemp = np.roll(Htemp, idx, axis=0)
    Htemp[:idx, :] = 0
    H = H + Htemp
# print(H)
S = np.roll(np.eye(ct.N), 1, axis=0)
S[0, :] = 0
# print(S)
x = cv.Variable(ct.N)
# obj = cv.Minimize(cv.norm(H * x - ct.y))
obj = cv.Minimize(cv.norm(cv.conv(ct.h, x)[:-3] - ct.y))
# cst = [S * x <= x]
cst = [0 <= x[0],
       x[:-1] <= x[1:]]
prb = cv.Problem(obj, cst)
prb.solve()
# print(prb.status)
pl.figure(figsize=(6, 6))
pl.plot(x.value, label=r'$x_{ml}$')
pl.plot(ct.xtrue, label=r'$x_{true}$')
prb = cv.Problem(obj)
prb.solve()
# print(prb.status)
pl.plot(x.value, label=r'$x_{ml,free}$')
pl.legend(loc='upper left', bbox_to_anchor=(1.03, 1.03), prop={'size': 18})
