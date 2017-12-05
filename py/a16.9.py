from IPython import get_ipython
get_ipython().magic('reset -sf')
print('\n'*100)
import numpy as np
import cvxpy as cv
import matplotlib as mp
#mp.use('QT4Agg')
import matplotlib.pyplot as pl
import pylab as py
import os, sys, time, copy#Λ
#haha=np.array([[0,1,2],
#               [3,4,5],
#               [6,7,8]])

tt = time.time()
#print('\ntime elapsed=', time.time()-tt)
#exec(open('ls_perm_meas_data.py').read())
#import ls_perm_meas_data as ct
#np.set_printoptions(precision=3)
#PTl[np.asmatrix([1,3,5]).T, [1,3,5]]
#legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))
import storage_tradeoff_data as ct

C = cv.Parameter()
D = cv.Parameter()
Q = cv.Parameter()
C.value = 3
D.value = 3
Q.value = 35

c = cv.Variable(ct.T)
q = cv.Variable(ct.T)

obj = cv.Minimize(ct.p.T * (ct.u + c))
CYC = np.eye(ct.T)
CYC[0, 0] = 0
CYC[-1, -1] = 0
CYC[np.asmatrix(np.r_[:ct.T-1]).T, np.r_[1:ct.T]] = np.eye(ct.T-1)
CYC[ct.T-1, 0] = 1
cst = [c <= C,
       -D <= c,
       0 <= q,
       q <= Q,
       CYC * q == q + c,
       ct.u + c >= 0]
prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status, prb.value)

f, ax1 = pl.subplots()
ax2 = ax1.twinx()
ax1.plot(ct.t, ct.u, label='u')
ax1.plot(ct.t, ct.p, label='p')
ax1.plot(ct.t, c.value, label='c')
ax2.plot(ct.t, q.value, label='q', color='m')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax2.legend(loc='center left', bbox_to_anchor=(1.05, .5))

nn = 200
Q_list = np.arange(nn)
cost = np.zeros(nn)
pl.figure(2)
for CD in [3, 1]:
    C.value = CD
    D.value = CD
    for idx in np.arange(nn):
        Q.value = Q_list[idx]
        prb.solve()
        if prb.status != 'optimal':
            input('...not optimal')
        cost[idx] = prb.value
    pl.plot(Q_list, cost, label='C=D='+str(CD))
pl.legend()


print('\ntime elapsed=', time.time()-tt)
