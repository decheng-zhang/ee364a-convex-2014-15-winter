file = 'simple_portfolio_data.py'
exec(open(file).read())

import cvxpy as cvx
import numpy as npy

x = cvx.Variable(n, 1)

obj = cvx.Minimize(cvx.quad_form(x, S))
cst = [pbar.T * x == npy.dot(pbar.T, x_unif),
        npy.ones((1, n)) * x == 1]
        #x >= 0,
        #npy.ones((1, n)) * cvx.max_elemwise(-x, 0) <= .5]
prb = cvx.Problem(obj, cst)
prb.solve()
print('a1. risk:', npy.sqrt(prb.value))
#print('portfolio\n', x.value)
cst = [pbar.T * x == npy.dot(pbar.T, x_unif),
        npy.ones((1, n)) * x == 1,
        x >= 0]
        #npy.ones((1, n)) * cvx.max_elemwise(-x, 0) <= .5]
prb = cvx.Problem(obj, cst)
prb.solve()
print('a2. risk:', npy.sqrt(prb.value))
#print('portfolio\n', x.value)
cst = [pbar.T * x == npy.dot(pbar.T, x_unif),
        npy.ones((1, n)) * x == 1,
        #x >= 0,
        npy.ones((1, n)) * cvx.max_elemwise(-x, 0) <= .5]
prb = cvx.Problem(obj, cst)
prb.solve()
print('a3. risk:', npy.sqrt(prb.value))
#print('portfolio\n', x.value)

print('check short constraint:',
npy.sum(
        npy.hstack(
            (-x.value, npy.zeros((n,1)))
            ).max(axis=1)
        )
)
print('uniform. risk:', npy.sqrt(x_unif.T * S * x_unif))
