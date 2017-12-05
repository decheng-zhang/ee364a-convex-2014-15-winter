import numpy as npy
import cvxpy as cvx
import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt

npy.random.seed(0)
(m, n) = (300, 100)
A = npy.random.rand(m, n)
A = npy.asmatrix(A)
b = A.dot(npy.ones((n, 1)))/2
b = npy.asmatrix(b)
c = -npy.random.rand(n, 1)
c = npy.asmatrix(c)

x = cvx.Variable(n, 1)
obj = cvx.Minimize(c.T * x)
cst = [A*x <= b,
        0 <= x,
        x <= 1]
prb = cvx.Problem(obj, cst)
prb.solve()
prb.status
L = prb.value
I = 100
t = npy.linspace(0, 1, I)
cx = npy.zeros(I)
mv = npy.zeros(I)
for idx in npy.arange(I):
    xh = x.value >= t[idx]
    xh = xh * 1
    cx[idx] = c.T * xh
    mv[idx] = max(A*xh - b)
plt.plot(t, cx, label = 'objective value')
plt.plot(t, mv, label = 'maximum violation')
cx[mv > 0]=float('inf')
U = min(cx)
tatU = t[min(npy.where(cx==U)[0])]
#t[cx[npy.isinf(cx)].size]# works
#t[cx.index(U)]# works with list but not ndarray
xlimi = plt.gca().get_xlim()
ylimi = plt.gca().get_ylim()
plt.plot(xlimi, [U, U], label = 'U')
plt.plot(xlimi, [L, L], label = 'L. U-L = ' '%.2f' % (U-L))
plt.plot(xlimi, [0, 0], label = '0')
plt.plot([tatU, tatU], ylimi,
        label = 't = %.2f' % tatU + '\n'
        't >= %.2f' % tatU + ': feasible\n'
        't < %.2f' % tatU + ': infeasible')

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1))
plt.xlabel('t')
plt.show()
