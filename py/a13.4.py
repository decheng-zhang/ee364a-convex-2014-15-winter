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

x = np.array([[.1],[.2],[-.05],[.1]])
S = cv.Semidef(4)
cst = [ S[0,0] == .2,
        S[1,1] == .1,
        S[2,2] == .3,
        S[3,3] == .1,
        S[0,1] >= 0,
        S[0,2] >= 0,
        S[1,2] <= 0,
        S[1,3] <= 0,
        S[2,3] >= 0]
obj = cv.Maximize(cv.quad_form(x, S))
prb = cv.Problem(obj, cst)
prb.solve()
#print(prb.status)
Sd = np.array([[.2,0,0,0],[0,.1,0,0],[0,0,.3,0],[0,0,0,.1]])
sd = np.dot(x.T, np.dot(Sd, x))
print('%-12s' % 'σ_wc', np.sqrt(x.T * S.value * x)[0,0])
print('%-12s' % 'σ w diag Σ', np.sqrt(sd[0,0]))
print('Σ\n', np.round(S.value, 3))



print('\ntime elapsed=', time.time()-tt)