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
# boolArray = [False] * 6
# sliceArray = PTl[np.asmatrix([1,3,5]).T, [1,3,5]]
# pl.figure(figsize=(6, 6))
# fig = pl.figure()
# ax = fig.gca(projection='3d')
# ax.plot(xs=p.value[0,:].A1,ys=p.value[1,:].A1,zs=p.value[2,:].A1)
# legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), prop={'size': 18})
# print(dir(ct))
# raise Exception('exit')
# print(cv.installed_solvers())

import rel_pwr_flow_data as ct
print(dir(ct))

g = cv.Variable(ct.k)
p = cv.Variable(ct.m)
pcont = cv.Variable(ct.m-1, ct.m)

obj = cv.Minimize(ct.c.T*g)
cst = [cv.abs(p) <= ct.Pmax.T,
       0 <= g,
       g <= ct.Gmax,
       ct.A * p == cv.vstack(-g, ct.d.T)]
prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status, 'cost', obj.value, 'gen', g.value.T)

cst += [ct.A*cv.vstack(pcont[:i, i], 0, pcont[i:, i]) == cv.vstack(-g, ct.d.T)
        for i in range(ct.m)]  # [:i, i] doesn't work on Pmax
cst += [cv.abs(pcont[:, i]) <= np.hstack((ct.Pmax[0, :i], ct.Pmax[0, i+1:])).T
        for i in range(ct.m)]
# note the difference
# cv.vstack(pcont[:i, i], 0, pcont[i:, i])
# np.hstack((ct.Pmax[0, :i], ct.Pmax[0, i+1:]))

prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status, 'cost', obj.value, 'gen', g.value.T)
