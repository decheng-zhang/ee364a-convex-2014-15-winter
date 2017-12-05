import numpy as npy
import cvxpy as cvx
import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt

A1 = npy.matrix('1, -.5; -.5 2')
A2 = npy.matrix('-1, 0')
A3 = npy.matrix('1, 2; 1, -4')
A4 = npy.matrix('5, 76')
x = cvx.Variable(2)
u = cvx.Parameter(2)
u.value = [-2, -3]

obj = cvx.Minimize(cvx.quad_form(x, A1) + A2*x)
cst = [A3*x <= u,
        A4*x <= 1]
prb = cvx.Problem(obj, cst)
prb.solve()
print(prb.status)
ps = prb.value
print('p*=', ps)
print('x1=', x.value[0, 0])
print('x2=', x.value[1, 0])
print('λ1=', cst[0].dual_value[0, 0])
print('λ2=', cst[0].dual_value[1, 0])
print('λ3=', cst[1].dual_value)
print('primal constraints:')
print(A3*x.value-u.value)
print(A4*x.value-1)
print('ok: primal constraints. dual constraints. complementary slackness')
print('dL/dx1=',
        2*x.value[0, 0]
        - x.value[1, 0]
        - 1
        + cst[0].dual_value[0, 0]
        + cst[0].dual_value[1, 0]
        + 5* cst[1].dual_value)
print('dL/dx2=',
        - x.value[0, 0]
        + 4* x.value[1, 0]
        + 2* cst[0].dual_value[0, 0]
        - 4* cst[0].dual_value[1, 0]
        + 76* cst[1].dual_value)
print('ok: ∇_x L vanishes')

u0 = [-2, -3]
de = [0, -.1, .1]
print('  δ1   δ2  p*_pred p*_exact  exact>=pred?')
lam= cst[0].dual_value
for idx in npy.arange(len(de)):
    for jdx in npy.arange(len(de)):
        u.value = [u0[0]+de[idx], u0[1]+de[jdx]]
        prb.solve()
        #print(u.value)
        #print(cst[0].dual_value)
        ps_pred = ps - ([de[idx], de[jdx]] * lam)[0,0]
        print('% 4.1g' % de[idx], '% 4.1g' % de[jdx], end = "")
        print('% 9.3f' % ps_pred, end = "")
        print('% 9.3f' % prb.value, end = "")
        #print(ps_pred-prb.value)
        print('% 10s' % (prb.value >= ps_pred))



