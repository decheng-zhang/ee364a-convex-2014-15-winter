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

tt = time.time()
#file = 'ls_perm_meas_data.py'
#exec(open(file).read())
import ls_perm_meas_data as ct

xvr = cv.Variable(ct.n, 1)

##this doesn't work because each pair of A[idx,:], y[jdx] corresponds to a
##completely different x*
#isychosen = [False] * ct.m
#bestjdx = [np.float("inf")] * ct.m#better than ones * inf: [idx] gives an array
#for idx in np.arange(ct.m):
#    bestresidual = np.float("inf")
#    for jdx in np.arange(ct.m):
#        if isychosen[jdx] == False:
#            prb = cv.Problem(cv.Minimize(\
#            cv.norm(ct.A[idx,:] * xvr - ct.y[jdx])))
#            prb.solve()
#            if prb.status != 'optimal':
#                sys.exit(235)
#            if prb.value < bestresidual:
#                bestresidual = prb.value
#                bestjdx[idx] = jdx
#    if bestjdx[idx] == np.float("inf"):
#        sys.exit(432)
#    isychosen[bestjdx[idx]] = True
#Pha = np.zeros((ct.m, ct.m))
#for idx in np.arange(ct.m):
#    Pha[bestjdx[idx], idx] = 1

##this doens't work. exhausted available fcns of cvxpy
#PTv = cv.Variable(ct.m, ct.m)
#cst = [ PTv <= 1,
#        0 <= PTv,
#        cv.norm(PTv, 1) <= ct.m,
#        cv.trace(PTv) >= (ct.m-ct.k),
#        cv.norm(PTv,'nuc') <= ct.m,
#        cv.sum_entries(PTv) == ct.m,
#        cv.sum_entries(PTv, axis = 0) == np.ones((1, ct.m)),
#        cv.sum_entries(PTv, axis = 1) == np.ones((ct.m, 1))]
#obj = cv.Minimize(cv.norm(ct.A * xvr - PTv * ct.y))
#prb = cv.Problem(obj, cst)
#prb.solve()
#Pha = copy.copy(PTv.value.T)
#for idx in np.arange(ct.m):
#    col = Pha[:, idx]
#    col = col.tolist()
#    ind = col.index(max(col))
#    Pha[:, idx] = 0
#    Pha[ind, :] = 0
#    Pha[ind, idx] = 1
#print('P var,', prb.status,
#      ', residual', '% .2e' % prb.value,
#      ', ||x-x_true||', '%.2e' % np.linalg.norm(xvr.value-ct.x_true))
#print('P hat,         '
#      ', residual', '%.2e' % np.linalg.norm(ct.A * xvr.value - Pha.T * ct.y))

##this doesn't work because reassignment should only be applied to outliers
#def construct_PT(np, ct, xvr, PTp):
#    '''construct PT by re-arranging eye'''
#    #sort A*x. sort y
#    #sort abs(sorted A*x - sorted y)
#    #start from the largest difference, changing PT to align A*x and y
#    #stop until trace drops below m-k
#    #print(np.abs(ycp[0]))
#    est = np.asarray(ct.A * xvr.value)
#    eid = est[:,0].argsort()[::-1]
#    yid = ct.y[:,0].argsort()[::-1]
#    dif = np.abs( est[eid] - ct.y[yid] )
#    did = dif[:,0].argsort()[::-1]
#    PTp.value = np.asmatrix(np.eye(ct.m))
#    for jdx in np.arange(ct.m):
#        #print('idx', idx, 'jdx', jdx,
#        #'eid', eid[did[jdx]], 'yid', yid[did[jdx]], end = ' ')
#        if eid[did[jdx]] != yid[did[jdx]]:
#            #look for 1 in target column
#            mov =\
#                np.matrix.tolist(
#                    np.squeeze(
#                        np.asarray(
#                            PTp.value[:, yid[did[jdx]]])))\
#                .index(1)
#            #look for 1 in target row
#            occ =\
#                np.matrix.tolist(
#                    np.squeeze(
#                        np.asarray(
#                            PTp.value[eid[did[jdx]], :])))\
#                .index(1)
#            PTp.value[mov, yid[did[jdx]]] = 0
#            PTp.value[eid[did[jdx]], occ] = 0
#            PTp.value[eid[did[jdx]], yid[did[jdx]]] = 1
#            PTp.value[mov, occ] = 1
#        else:
#            if PTp.value[yid[did[jdx]], yid[did[jdx]]] == 0:
#                input('...fishy')
#        det = np.abs(np.linalg.det(PTp.value))
#        #print('det', det)
#        if det != 1:
#            input('...det not 1')
#        if np.trace(PTp.value) <= ct.m - ct.k:
#            break
#
def printout(name, prbv, np, x, ct):
    print('%-10s' % name, ', residual %5.4g' % prbv,
      ', ||x-x_true|| %5.3g' % np.linalg.norm(x - ct.x_true))
      #(np.linalg.norm(x - ct.x_true)/np.linalg.norm(ct.x_true)*100), '%')
#
prb = cv.Problem(cv.Minimize(cv.norm(ct.A * xvr - ct.y)))
prb.solve()
if prb.status != 'optimal':
    input('...not optimal')
printout('P = I', prb.value, np, xvr.value, ct)
#
prb = cv.Problem(cv.Minimize(cv.norm(ct.A * xvr - ct.y_true)))
prb.solve()
if prb.status != 'optimal':
    input('...not optimal')
printout('true y', prb.value, np, xvr.value, ct)
#
def construct_PT(np, ct, xvr, PTp):
    #rank residuals
    rid = np.asarray(np.abs(ct.A * xvr.value - PTp.value * ct.y))\
            [:,0].argsort()[::-1]
    #re-fit with good ones
    kid = rid[ct.k:]
    cv.Problem(cv.Minimize(\
        cv.norm(ct.A[kid,:] * xvr - PTp[kid,:] * ct.y))).solve()
    #permute the bad ones
    pid = rid[0:ct.k]#this doesn't include kth!!!!!!
    eid = np.asarray(ct.A[pid,:] * xvr.value)[:,0].argsort()[::-1]
    yid = np.asarray(PTp.value[pid,:] * ct.y)[:,0].argsort()[::-1]
    PTs = np.zeros((ct.k,ct.k))
    for jdx in np.arange(ct.k):
        PTs[eid[jdx],yid[jdx]] = 1
    #print(PTs)
    #if np.trace(PTs) == ct.k:#this can be used to exit
    #    print('no permute')
    PTl = np.eye(ct.m)
    PTl[np.asmatrix(pid).T, pid] = PTs
    #[pid,pid], [[pid],pid] don't work!!!!!!
    #http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar
    PTp.value = PTl * PTp.value

    det = np.abs(np.linalg.det(PTp.value))
    #print('det', det)
    if det != 1:
        input('...det not 1')
#
PTp = cv.Parameter(ct.m, ct.m)
PTp.value = np.asmatrix(np.eye(ct.m))
prbr = cv.Problem(cv.Minimize(\
        cv.sum_entries(cv.huber(ct.A * xvr - PTp * ct.y))))
prb = cv.Problem(cv.Minimize(cv.norm(ct.A * xvr - PTp * ct.y)))
bestres = np.float("inf")
bestxer = np.float("inf")
bestidx = np.float("inf")
bestP = np.float("inf")
#
pl.figure(1)
for idx in np.arange(8):
    #rough fit
    prbr.solve()
    #rearrange
    construct_PT(np, ct, xvr, PTp)
    #tight fit
    prb.solve()
    if prb.status != 'optimal':
        input('...not optimal')
    #print('%.2e' % prb.value)
    #pl.plot(idx, prb.value, marker = ".")
    xertry = np.linalg.norm(xvr.value-ct.x_true)
    pl.plot(idx, xertry, marker = ".")
    #pl.plot(np.abs(ct.A * xvr.value - PTp.value * ct.y), marker='.')
    #pl.show()
    if xertry < bestxer:
        bestres = prb.value
        bestidx = idx
        bestxer = xertry
        bestP = PTp.value.T
        bestx = xvr.value
    #pl.figure(idx)
    #py.imshow(PTp.value.T, interpolation = 'nearest', clim = (0, 1), cmap = 'gray')
    #pl.show()

printout('P var end', prb.value, np, xvr.value, ct)
printout('P var best', bestres, np, bestx, ct)
print('at idx=', bestidx)

#
pl.figure(100)

pl.subplot(1, 2, 1)
py.imshow(ct.P, interpolation = 'nearest', clim = (0, 1), cmap = 'gray')
py.title('P true')

pl.subplot(1, 2, 2)
py.imshow(bestP, interpolation = 'nearest', clim = (0, 1), cmap = 'gray')
py.title(r'$\hat{P}$ best')

pl.figure(101)

pl.subplot(1, 2, 1)
py.imshow(bestP*bestP.T, interpolation = 'nearest', clim = (0, 1), cmap = 'gray')
py.title(r'$\hat{P}\hat{P}^T$')

pl.subplot(1, 2, 2)
py.imshow(PTp.value.T, interpolation = 'nearest', clim = (0, 1), cmap = 'gray')
py.title(r'$\hat{P}$ end')

#py.grid(True)
pl.show()

print('\ntime elapsed=', time.time()-tt)
