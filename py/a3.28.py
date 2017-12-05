# from IPython import get_ipython
# get_ipython().magic('reset -sf')
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
# import ls_perm_meas_data as ct
np.set_printoptions(precision=3)
# PTl[np.asmatrix([1,3,5]).T, [1,3,5]]
# legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))

nn = 16  # 2**4

vo = np.asmatrix([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
v1 = np.asmatrix([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
v2 = np.asmatrix([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
v3 = np.asmatrix([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
v4 = np.asmatrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
v51 = np.asmatrix([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
v52 = np.asmatrix([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
vu = np.asmatrix([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
pp = cv.Variable(nn, 1)

obj = [cv.Minimize(vo * pp),
       cv.Maximize(vo * pp)]
tag = ['min p4', 'max p4']
for idx in np.arange(2):
    cst = [  # pp <= 1,
           0 <= pp,
           vu * pp == 1,
           v1 * pp == .9,
           v2 * pp == .9,
           v3 * pp == .1,
           v4 * pp == .7 * v3 * pp,
           v52 * pp == .6 * v51 * pp]
    prb = cv.Problem(obj[idx], cst)
    prb.solve()
    print(prb.status)
    print(tag[idx], prb.value)
    # print(pp.value)
