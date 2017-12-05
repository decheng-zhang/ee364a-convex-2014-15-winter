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

import veh_speed_sched_data as ct

lowDiag = np.tril(np.ones((ct.n, ct.n)))
u = cv.Variable(ct.n)
obj = cv.Minimize(cv.sum_entries(
        cv.mul_elemwise(ct.d, ct.a*cv.inv_pos(u)+ct.b+ct.c*u)))
cst = [ct.tau_min <= lowDiag * cv.mul_elemwise(ct.d, u),
       lowDiag * cv.mul_elemwise(ct.d, u) <= ct.tau_max,
       1/ct.smin >= u,
       u >= 1/ct.smax]
prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status)
print('fuel', prb.value)

dAccu = lowDiag * ct.d

pl.step(np.vstack((0, dAccu)), np.vstack((0, 1/u.value)))
