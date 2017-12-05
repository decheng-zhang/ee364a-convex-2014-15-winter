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

import spacecraft_landing_data as ct

f = cv.Variable(3, ct.K+1)
v = cv.Variable(3, ct.K+1)
p = cv.Variable(3, ct.K+1)
# obj = cv.Minimize(ct.gamma * cv.norm(f, "nuc") * ct.h)  # optimal_inaccurate
obj = cv.Minimize(ct.gamma * cv.sum_entries(cv.norm(f, axis=0)) * ct.h)

hgm = np.zeros((3, ct.K))
hgm[2, :] = np.ones((1, ct.K)) * ct.h*ct.g
cst = [v[:, 0] == ct.v0,
       p[:, 0] == ct.p0,
       # f[:, 0] == 0,  # careful, 0 and 1 indexing
       v[:, 1:] == v[:, :-1] + ct.h/ct.m*f[:, :-1] - hgm,
       p[:, 1:] == p[:, :-1] + ct.h/2*(v[:, :-1] + v[:, 1:]),
       v[:, ct.K] == 0,
       p[:, ct.K] == 0,
       p[2, :] >= cv.norm(p[:2, :], axis=0) * ct.alpha,
       ct.Fmax >= cv.norm(f, axis=0)]
prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status)
print('fuel', prb.value)

ct.ax.plot(xs=p.value[0, :].A1, ys=p.value[1, :].A1, zs=p.value[2, :].A1)
# matrix.A1 gives array
# ct.ax.quiver(p.value[0, :].A1, p.value[1, :].A1, p.value[2, :].A1,
#             f.value[0, :].A1, f.value[1, :].A1, f.value[2, :].A1,
#             length=10)
# as of 201612, arrows of quiver are all of the same length

###########################
Kbutt = 1  # infeasible
Khead = ct.K  # works

while True:
    ct.K = int(np.ceil((Khead + Kbutt)/2))
    f = cv.Variable(3, ct.K+1)
    v = cv.Variable(3, ct.K+1)
    p = cv.Variable(3, ct.K+1)
    # obj = cv.Minimize(1)  # doesn't change eventual K
    obj = cv.Minimize(ct.gamma * cv.sum_entries(cv.norm(f, axis=0)) * ct.h)

    hgm = np.zeros((3, ct.K))
    hgm[2, :] = np.ones((1, ct.K)) * ct.h*ct.g
    cst = [v[:, 0] == ct.v0,
           p[:, 0] == ct.p0,
           # f[:, 0] == 0,  # careful, 0 and 1 indexing
           v[:, 1:] == v[:, :-1] + ct.h/ct.m*f[:, :-1] - hgm,
           p[:, 1:] == p[:, :-1] + ct.h/2*(v[:, :-1] + v[:, 1:]),
           v[:, ct.K] == 0,
           p[:, ct.K] == 0,
           p[2, :] >= cv.norm(p[:2, :], axis=0) * ct.alpha,
           ct.Fmax >= cv.norm(f, axis=0)]
    prb = cv.Problem(obj, cst)
    prb.solve()
    print(ct.K, prb.status, Kbutt, Khead)
    if prb.status == 'optimal':
        Khead = ct.K
    else:
        Kbutt = ct.K
    if Khead <= Kbutt + 1 and prb.status == 'optimal':
        break

print('fuel', (ct.gamma * cv.sum_entries(cv.norm(f, axis=0)) * ct.h).value)

ct.ax.plot(xs=p.value[0, :].A1, ys=p.value[1, :].A1, zs=p.value[2, :].A1)

pl.show()
print('\ntime elapsed=', time.time()-tt)
