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
#file = 'ls_perm_meas_data.py'
#exec(open(file).read())
#import ls_perm_meas_data as ct



p = np.array([.5, .6, .6, .6, .2])
q = np.array([10, 5, 5, 20, 10])
A = np.array([[1, 1, 0, 0, 0],
              [0, 0, 0, 1, 0],
              [1, 0, 0, 1, 1],
              [0, 1, 0, 0, 1],
              [0, 0, 1, 0, 0]]).T
x = cv.Variable(5)
obj = cv.Minimize(cv.max_entries(A * x) - p.T * x)
cst = [x <= q, 0 <= x]
prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status, prb.value, x.value.T)

t = cv.Variable()
obj = cv.Minimize(t - p.T * x)
cst = [A * x <= t, x <= q, 0 <= x]
prb = cv.Problem(obj, cst)
prb.solve()
print(prb.status, prb.value, x.value.T)

np.set_printoptions(precision=3)

print(cst[0].dual_value.T)
print(cst[1].dual_value.T)
print(cst[2].dual_value.T)

print('wc house profit % 10.2f' % -prb.value)
print('wc house profit, x=q % 3.2f' % -(np.max(np.dot(A, q)) - np.dot(p, q)))
print('imputed probabilities:', np.asarray(cst[0].dual_value)[:,0])

print('\ntime elapsed=', time.time()-tt)