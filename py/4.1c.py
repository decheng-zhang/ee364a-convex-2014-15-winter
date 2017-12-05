import cvxpy as cvx

# Create two scalar optimization variables.
x1 = cvx.Variable()
x2 = cvx.Variable()

# Create two constraints.
constraints = [2*x1 +   x2 >= 1,
                 x1 + 3*x2 >= 1,
                 x1        >= 0,
                 x2        >= 0]

# Form objective.
obj = cvx.Minimize(x1)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x1.value, x2.value)
