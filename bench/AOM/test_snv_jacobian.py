import numpy as np

def snv(X):
    m = np.mean(X, axis=1, keepdims=True)
    s = np.std(X, axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s

X = np.random.randn(10, 100)
c = np.random.randn(100)

# Numerical Jacobian
eps = 1e-5
J_num = np.zeros((100, 100))
x0 = X[0].copy()
for i in range(100):
    x_plus = x0.copy()
    x_plus[i] += eps
    x_minus = x0.copy()
    x_minus[i] -= eps
    J_num[:, i] = (snv(x_plus.reshape(1, -1))[0] - snv(x_minus.reshape(1, -1))[0]) / (2 * eps)

# Analytical Jacobian
m = np.mean(x0)
s = np.std(x0)
S_x = (x0 - m) / s
J_ana = (np.eye(100) - np.ones((100, 100)) / 100 - np.outer(S_x, S_x) / 100) / s

print("Jacobian diff:", np.max(np.abs(J_num - J_ana)))

# Adjoint
adj_num = J_num.T @ c
c_mean = np.mean(c)
dot_prod = np.dot(S_x, c) / 100
adj_ana = (c - c_mean - dot_prod * S_x) / s

print("Adjoint diff:", np.max(np.abs(adj_num - adj_ana)))
