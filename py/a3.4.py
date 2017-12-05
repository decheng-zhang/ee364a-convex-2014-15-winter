import cvxpy as cvx
import numpy as npy
import matplotlib as mpl
mpl.use('QT4Agg') #stackoverflow.com/questions/4930524/
#mpl.use('TKAgg') #stackoverflow.com/questions/24372578/
import matplotlib.pyplot as plt

m = 5
n = 4
A = npy.matrix('1 2 0 1;\
                0 0 3 1;\
                0 3 1 1;\
                2 1 2 5;\
                1 0 3 2')
cmax = npy.ones((m, 1)) * 100;
p = npy.matrix('3; 2; 7; 6')
pdisc = npy.matrix('2; 1; 4; 2')
q = npy.matrix('4; 10; 5; 10')
pq = npy.multiply(p, q);

x = cvx.Variable(n, 1);
t = cvx.Variable(n, 1);
cst = [ x >= 0,
        A*x <= cmax,
        cvx.mul_elemwise(p, x) >= t,
        pq+cvx.mul_elemwise(pdisc, x - q) >= t]

obj = cvx.Maximize(npy.ones((1, n)) * t)

prb = cvx.Problem(obj, cst)

prb.solve()
print(  prb.status, '\n',
        'activity levels x\n', x.value, '\n',
        'revenues t\n', t.value, '\n',
        'total ', prb.value, '\n',
        'avg price t/x \n', npy.divide( t.value, x.value))

