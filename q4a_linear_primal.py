import numpy as np
from scipy.optimize import minimize


def primal_objective(w_flat, X, Y, C, n, d, k):
    W = w_flat[:d * k].reshape(d, k)
    b = w_flat[d * k:]
    F = X @ W + b
    viol = np.maximum(0.0, 1.0 - Y * F)
    loss = C * np.sum(viol * viol)
    reg = 0.5 * np.sum(W * W)
    G_viol = -2.0 * C * (viol * Y)
    gW = W + X.T @ G_viol
    gb = G_viol.sum(axis=0)
    return reg + loss, np.concatenate([gW.ravel(), gb])


def solve_linear_primal(train_x, train_y, C):
    n, d = train_x.shape
    k = 10
    Y = np.full((n, k), -1.0)
    Y[np.arange(n), train_y] = 1.0
    w0 = np.zeros(d * k + k)
    res = minimize(primal_objective, w0, args=(train_x, Y, C, n, d, k),
                   method="L-BFGS-B", jac=True,
                   options={"maxiter": 400, "maxcor": 20, "maxfun": 500})
    W = res.x[:d * k].reshape(d, k)
    b = res.x[d * k:]
    return W, b


def linear_scores(x, W, b):
    return x @ W + b
