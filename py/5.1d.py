import numpy as npy
import cvxpy as cvx
import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt

def p_fcn(u):
    tmp = npy.zeros(u.shape)
    for idx in npy.arange(u.size):
        if u[idx] <= -1:
            tmp[idx] = float("-inf")
        elif u[idx] <= 8:
            tmp[idx] = 11 + u[idx] - 6 * npy.sqrt(1+u[idx])
        else:
            tmp[idx] = 1
    return tmp

N = 100
u = npy.linspace(-2, 10, num = N)
p = p_fcn(u)
plt.plot(u, p, label = 'p*(u)')
xli = plt.gca().get_xlim();
plt.plot(0, 5, marker = 'o', label = 'p*(0)')
plt.xlabel('u')
plt.legend()
plt.show()
