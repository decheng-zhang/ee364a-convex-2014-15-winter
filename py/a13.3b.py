import cvxpy as cvx
import numpy as npy
import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt

file = 'simple_portfolio_data.py'
exec(open(file).read())

nn = 20;
mus = npy.logspace(-1, 5, num = nn)
avg = npy.zeros((nn, 1))
std = npy.zeros((nn, 1))
han = []

mu = cvx.Parameter(sign='positive')
x = cvx.Variable(n, 1)
exp = pbar.T * x;
var = cvx.quad_form(x, S)
obj = cvx.Minimize(-exp + mu * var)
cs1 = [npy.ones((1, n)) * x == 1]
prb = cvx.Problem(obj, [])
for idx in range(2):
    if idx == 0:
        prb.constraints = cs1 + [x >= 0]
        lab = 'long only'
    else:
        prb.constraints = cs1 + \
                [npy.ones((1, n)) * cvx.max_elemwise(-x, 0) <= .5]
        lab = 'total short <= .5'
    for jdx in range(nn):
        mu.value = mus[jdx]
        prb.solve()
        #print(prb.status)
        avg[jdx] = exp.value
        std[jdx] = npy.sqrt(var.value)
    #plot
    hantemp = plt.plot(std, avg, label = lab)
    han = han + hantemp
#plt.gca().set_aspect('equal')
ylimi = plt.gca().get_ylim()
plt.gca().set_ylim((0, ylimi[1]))
plt.ylabel('mean')
plt.xlabel('standard deviation')
plt.legend(handles = han, loc = 'lower right')
plt.show()

