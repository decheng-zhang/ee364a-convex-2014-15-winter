import numpy as npy
import cvxpy as cvx
import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt

N = 300
x = npy.linspace(-1, 5, N)
f0 = x**2 + 1

plt.figure(1)
plt.plot(x, f0, linewidth = 3, label = 'f' r'$_0$')
#plt.gca().set_aspect('equal')

l_list = npy.logspace(-1, 1, endpoint = True)#npy.arange(1,4)

def L_fcn(l, x):
    """l is a number. x is an array"""
    return (l+1) * x**2 - 6*l*x + 8*l+1

Lmin_list = npy.zeros([l_list.size, 2])
for idx in npy.arange(l_list.size):
    L = L_fcn(l_list[idx], x)
    #print(L)
    hand, = plt.plot(x, L, color = 'g')
    if idx == l_list.size - 1:
        #hand.label = 'L with different ' r'$\lambda$'
        #^doesn't work!
        hand.set_label('L with different ' r'$\lambda$' '\'s')
    Lmin_list[idx, 1] = min(L)
    Lmin_list[idx, 0] = x[list(L).index(Lmin_list[idx, 1])]

plt.plot(Lmin_list[:,0], Lmin_list[:,1],
            marker = '+', color = 'r',
            label = 'min(L) for different ' r'$\lambda$' '\'s')

plt.gca().set_ylim((-5, 20))
yli = plt.gca().get_ylim()
plt.plot(npy.array([2, 2]), yli, color = 'm')
plt.plot(npy.array([4, 4]), yli, color = 'm', label = 'edges of feasible range')

pstar = 5
xstar = 2
plt.plot(xstar, pstar, marker = 'o')
xli = plt.gca().get_xlim()
plt.plot(xli, pstar*npy.array([1, 1]), color = 'c', label = 'p*')
plt.xlabel('x')
plt.legend(loc = 2)






l = npy.linspace(-1, 10)

def g_fcn(l):
    return (-l**2+9*l+1)/(l+1)



g = g_fcn(l)
plt.figure(2)
plt.plot(l, g, label = 'g')
plt.xlabel(r'$\lambda$')
plt.legend()
plt.show()

