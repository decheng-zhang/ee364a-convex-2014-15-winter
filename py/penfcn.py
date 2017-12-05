import numpy as npy
import cvxpy as cvx
import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt


"""def f0(u):
    return npy.abs(u)
def f1(u):
    return u**2
def f2(u, a):
    #u is an 1D array. a is a positive number
    t = npy.absolute(u) - a
    t = npy.vstack((t, npy.zeros(t.shape)))
    return npy.max(t, axis = 0)
def f3(u):
    return -log(1-u**2)
"""
m = 100
n = 30

#npy.random.seed(1)
A = npy.random.random((m, n)) * 2 - 1
x_known = npy.random.random((n, 1)) * 2 - 1
v = npy.random.random((m, 1)) * 2 - 1
b = npy.dot(A, x_known) + v

r = cvx.Variable(m, 1)
x = cvx.Variable(n, 1)
z = npy.zeros((n, 1))
a = .25
cst = [r == A * x - b]
f, (ax) = plt.subplots(4, sharex = True)
bins = npy.linspace(-2,2,81)+.025
bins = bins[:-1]
for idx in [0,1,2,3]:
    if idx == 0:
        fcn = cvx.sum_entries(cvx.abs(r))
    elif idx == 1:
        fcn = cvx.sum_entries(cvx.square(r))
    elif idx == 2:
        fcn = cvx.sum_entries(
            cvx.max_entries(
                cvx.hstack(cvx.abs(r)-a, npy.zeros((m,1))),
                axis = 1))
    elif idx == 3:
        fcn = cvx.sum_entries(-cvx.log(1-cvx.square(r)))
    else:
        print('bad')


    print(idx)
    prb = cvx.Problem(cvx.Minimize(fcn), cst)
    prb.solve()
    print(prb.status)
    ax[idx].hist(r.value, bins)
plt.show()
    
    
