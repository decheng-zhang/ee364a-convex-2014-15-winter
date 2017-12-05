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

import blend_design_data as ct
# print(dir(ct))


def monofit(cv, n, logW1, logP):
    aa = cv.Variable(n+1, 1)
    aTlogW1 = aa.T * logW1
    obj = cv.Minimize(cv.norm(aTlogW1 - logP, 2))
    cst = [aTlogW1 <= logP]  # wanted to do consecutive fit, all summed
    prb = cv.Problem(obj, cst)
    prb.solve()
    # print(prb.status, prb.value)
    return aa


def intersection(aa, bb):
    '''aa and bb are tuples, representing intervals in R'''
    if aa[0] > aa[1] or bb[0] > bb[1]:
        print('intersection has problem')
        while True:
            pass
    if aa[1] >= bb[0] and aa[0] <= bb[1]:
        return (max(aa[0], bb[0]), min(aa[1], bb[1]))
    else:
        return (np.NaN, np.NaN)


def find_interval(pp, pt):
    '''pp is tuple, interval. pp[0]<=pp[1]. pt is R'''
    if pp[0] > pp[1]:
        print('find_interval has problem')
        while True:
            pass
    if pt >= pp[1]:
        return pp
    if pt <= pp[0]:
        return (np.NaN, np.NaN)
    return (pp[0], pt)


def comp_spec(aa, sp):
    '''aa is unordered tuple. sp is spec. output tuple
    eg: aa=1, 9; sp = 3; output is 0, .25
    eg: aa=9, 1; sp = 3; output is .75, 1
    eg: aa=1, 9; sp = 9; output is 1, 1
    eg: aa=9, 1; sp = 9; output is 0, 0
    eg: aa=1, 9; sp = 1; output is 0, 0
    eg: aa=9, 1; sp = 1; output is 1, 1
    eg: aa=1, 9; sp = 0; output is nan, nan
    eg: aa=9, 1; sp = 0; output is nan, nan
    eg: aa=1, 9; sp = 10; output is 0, 1
    eg: aa=9, 1; sp = 10; output is 0, 1'''
    if sp > max(aa):
        return (0, 1)
    if sp < min(aa):
        return (np.NaN, np.NaN)
    rr = (sp-aa[0])/(aa[1]-aa[0])
    if aa[0] <= aa[1]:
        return (0, rr)
    else:
        return (rr, 1)

logW = np.log(ct.W)
logW1 = np.vstack((logW, np.ones((1, ct.k))))
logWm = np.log(ct.W_min)
logWM = np.log(ct.W_max)
logP = np.log(ct.P)
logD = np.log(ct.D)
logA = np.log(ct.A)
logPs = np.log(ct.P_spec)
logDs = np.log(ct.D_spec)
logAs = np.log(ct.A_spec)

# approach 1 doens't work

ap = monofit(cv, ct.n, logW1, logP)
ad = monofit(cv, ct.n, logW1, logD)
aa = monofit(cv, ct.n, logW1, logA)
lgW1 = cv.vstack(cv.Variable(ct.n, 1), 1)
cst = [ap.value.T * lgW1 <= logPs,
       ad.value.T * lgW1 <= logDs,
       aa.value.T * lgW1 <= logAs,
       lgW1[:-1] <= logWM,
       logWm <= lgW1[:-1]]
prb = cv.Problem(cv.Minimize(0), cst)
prb.solve()
# print(prb.status, prb.value)
# print('W by affine fit', np.exp(lgW1.value[:-1].T))


'''
print('fitting check\n',
      ap.value.T * logW1, '\n', logP, '\n',
      ad.value.T * logW1, '\n', logD, '\n',
      aa.value.T * logW1, '\n', logA)
print('feasibility check\n',
      ap.value.T * lgW1.value, logPs, '\n',
      ad.value.T * lgW1.value, logDs, '\n',
      aa.value.T * lgW1.value, logAs, '\n',)
'''

'''
print('log P spec %.2f' % logPs, '\n', np.vstack((logP, logP <= logPs)), '\n',
      'log D spec %.2f' % logDs, '\n', np.vstack((logD, logD <= logDs)), '\n',
      'log A spec %.2f' % logAs, '\n', np.vstack((logA, logA <= logAs)), '\n',)
print('log W min', logWm, 'max %.2f' % logWM)
print(logW)
'''

# approach 2

# print(intersection((1, 9), (4, 6)))
# print(find_interval((3, 8), 7))

logP = np.squeeze(np.asarray(logP))
logD = np.squeeze(np.asarray(logD))
logA = np.squeeze(np.asarray(logA))

for idx in range(ct.k):
    for jdx in np.arange(idx+1, ct.k):
        # print(idx, jdx, comp_spec((logP[idx], logP[jdx]), logPs))
        # print(idx, jdx, comp_spec((logD[idx], logD[jdx]), logDs))
        # print(idx, jdx, comp_spec((logA[idx], logA[jdx]), logAs))
        inP = comp_spec((logP[idx], logP[jdx]), logPs)
        inD = comp_spec((logD[idx], logD[jdx]), logDs)
        inA = comp_spec((logA[idx], logA[jdx]), logAs)
        tar = intersection(intersection(inP, inD), inA)
        if ~np.isnan(tar[0]):
            rr = (tar[0] + tar[1])/2
            tglgW = logW[:, idx] * (1-rr) + logW[:, jdx] * rr  # ha. careful
            # print(idx, jdx)
            # print(tar, rr)
            print('desired W', np.exp(tglgW.T))

# approach 3

th = cv.Variable(ct.k)
cst = [logP * th <= logPs,
       logD * th <= logDs,
       logA * th <= logAs,
       th >= 0,
       sum(th) == 1]
obj = cv.Minimize(0)
prb = cv.Problem(obj, cst)
prb.solve()
# print(prb.status)
# print('th', th.value.T)
print('desired W', np.exp(logW * th.value).T)
