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

import fba_data as ct

v = cv.Variable(ct.n)
obj = cv.Maximize(v[-1])
cst = [ct.S*v == 0,
       0 <= v,
       v <= ct.vmax]
prb = cv.Problem(obj, cst)
prb.solve()
Gstar = v[-1].value
print(prb.status)
print('max cell growth rate', Gstar)
# print(cst[0].dual_value)
print('lagrange multiplier of "v <= vmax"')
print(cst[2].dual_value.T)
print('lagrange multiplier of "v <= vmax" > .1 ?')
print(cst[2].dual_value.T > .1)

isEssentialIdx = []
notEssentialIdx = []
for idx in np.arange(ct.n):
    ###############################################
    vmax = copy.copy(ct.vmax)
    ###############################################
    vmax[idx] = 0
    cst[-1] = v <= vmax
    prb = cv.Problem(obj, cst)
    prb.solve()
    # print(prb.status)
    if v[-1].value < .2*Gstar:
        isEssentialIdx += [idx]
    else:
        notEssentialIdx += [idx]
print('isEssentialIdx:', isEssentialIdx)

synLethalIdx = []
for idx in np.arange(len(notEssentialIdx)):
    for jdx in np.r_[idx+1: len(notEssentialIdx)]:
        ###############################################
        vmax = copy.copy(ct.vmax)
        ###############################################
        vmax[notEssentialIdx[idx]] = 0
        vmax[notEssentialIdx[jdx]] = 0
        cst[-1] = v <= vmax
        prb = cv.Problem(obj, cst)
        prb.solve()
        # print(notEssentialIdx[idx], notEssentialIdx[jdx], prb.status)

        if v[-1].value < .2*Gstar:                                  ##########
            synLethalIdx += (notEssentialIdx[idx], notEssentialIdx[jdx]),
                                                                    ##########
print('synLethalIdx', synLethalIdx)
