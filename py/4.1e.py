import matplotlib as mpl
mpl.use('QT4Agg') #stackoverflow.com/questions/4930524/  ctrl+W closes fig
#mpl.use('TKAgg') #stackoverflow.com/questions/24372578/
import matplotlib.pyplot as plt
import numpy as npy
import cvxpy as cvx

x1 = cvx.Variable()
x2 = cvx.Variable()

constraints = [2*x1 +   x2 >= 1,
                 x1 + 3*x2 >= 1,
                 x1        >= 0,
                 x2        >= 0]

obj = cvx.Minimize(cvx.square(x1)+9*cvx.square(x2))

prob = cvx.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x1.value, x2.value)

x1=npy.linspace(0,1)
x2=1-2*x1
plt.plot(x1,x2)

x2=(1-x1)/3
plt.plot(x1,x2)

th=npy.linspace(0,npy.pi/2)
a=npy.sqrt(13/25)
b=a/3
x1=a*npy.cos(th)
x2=b*npy.sin(th)
plt.plot(x1,x2)

a=npy.sqrt(1/2)
b=a/3
x1=a*npy.cos(th)
x2=b*npy.sin(th)
plt.plot(x1,x2)


#plt.ylabel('some numbers')
plt.gca().set_aspect('equal')#, adjustable='box')
plt.gca().set_ylim(0,1)

#plt.draw()
#plt.ion()
plt.show()
#plt.show(block=False);input();plt.close('all')


